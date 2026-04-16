import asyncio
import collections
import html
import json
import logging
import numbers
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5, sha256
from typing import Any, Literal, Union

import numpy as np
import tiktoken
from pydantic import BaseModel

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger("nano-graphrag")
logging.getLogger("neo4j").setLevel(logging.ERROR)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}..."
                        )
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None


def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if "." in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist


def extract_values_from_json(
    json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False
):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group("key").strip('"')  # Strip quotes from key
        value = match.group("value").strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")

    return extracted_values


def convert_response_to_json(response: Union[str, BaseModel]) -> dict:
    """Convert response to JSON dict, handling both strings and pydantic models."""
    # If response is already a pydantic model, convert to dict
    if isinstance(response, BaseModel):
        return response.model_dump(by_alias=False)

    # Otherwise, treat as string and parse
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")

    return prediction_json


class TokenizerWrapper:
    _MAX_CACHE = 8192

    def __init__(
        self,
        tokenizer_type: Literal["tiktoken", "huggingface"] = "tiktoken",
        model_name: str = "gpt-4o",
    ):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self._tokenizer = None
        self._encode_cache: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
        self._decode_cache: collections.OrderedDict[tuple, str] = collections.OrderedDict()
        self._lazy_load_tokenizer()

    def _lazy_load_tokenizer(self):
        if self._tokenizer is not None:
            return
        logger.info(f"Loading tokenizer: type='{self.tokenizer_type}', name='{self.model_name}'")
        if self.tokenizer_type == "tiktoken":
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        elif self.tokenizer_type == "huggingface":
            if AutoTokenizer is None:
                raise ImportError(
                    "`transformers` is not installed. Please install it via `pip install transformers` to use HuggingFace tokenizers."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        else:
            raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")

    def get_tokenizer(self):
        """Provides access to the underlying tokenizer object."""
        self._lazy_load_tokenizer()
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        self._lazy_load_tokenizer()
        if text in self._encode_cache:
            self._encode_cache.move_to_end(text)
            return self._encode_cache[text]
        result = self._tokenizer.encode(text)
        if len(self._encode_cache) >= self._MAX_CACHE:
            self._encode_cache.popitem(last=False)
        self._encode_cache[text] = result
        return result

    def decode(self, tokens: list[int]) -> str:
        self._lazy_load_tokenizer()
        key = tuple(tokens)
        if key in self._decode_cache:
            self._decode_cache.move_to_end(key)
            return self._decode_cache[key]
        result = self._tokenizer.decode(tokens)
        if len(self._decode_cache) >= self._MAX_CACHE:
            self._decode_cache.popitem(last=False)
        self._decode_cache[key] = result
        return result

    def decode_batch(self, tokens_list: list[list[int]]) -> list[str]:
        self._lazy_load_tokenizer()
        # Defensive: simulating list concatenation via newline
        if self.tokenizer_type == "tiktoken":
            return [self._tokenizer.decode(tokens) for tokens in tokens_list]
        elif self.tokenizer_type == "huggingface":
            return self._tokenizer.batch_decode(tokens_list, skip_special_tokens=True)
        else:
            raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")


def truncate_list_by_token_size(
    list_data: list, key: callable, max_token_size: int, tokenizer_wrapper: TokenizerWrapper
):
    """Truncate a list of data by token size using a provided tokenizer wrapper."""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += (
            len(tokenizer_wrapper.encode(key(data))) + 1
        )  # Defensive: simulating list concatenation via newline
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def compute_sha256_id(content, prefix: str = ""):
    return prefix + sha256(content.encode("utf-8")).hexdigest()


def generate_stable_entity_id(
    entity_name: str,
    entity_type: str = "entity",
    namespace: str = "default",
):
    # Entity ID is based only on namespace + entity_name, not entity_type
    # This ensures stable IDs across re-indexing when entity types change
    normalized = f"{namespace}:{entity_name.strip().lower()}"
    return compute_sha256_id(normalized, prefix="entity_")


def generate_stable_relationship_id(
    src_entity_id: str,
    tgt_entity_id: str,
    relation_type: str = "related",
):
    left, right = sorted([src_entity_id, tgt_entity_id])
    normalized = f"{left}|{right}|{relation_type.strip().lower()}"
    return compute_sha256_id(normalized, prefix="rel_")


def pack_user_ass_to_openai_messages(
    prompt: str, generated_content: str, using_amazon_bedrock: bool
):
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generated_content},
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    def final_decro(func):
        semaphore = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def serialize_source_ids(source_ids: list[str]) -> str:
    """Serialize source IDs to a JSON array string."""
    return json.dumps(sorted(set(source_ids)))


def deserialize_source_ids(source_id_str: str) -> list[str]:
    """Deserialize source IDs from JSON array string, with fallback to <SEP> parsing."""
    if not source_id_str:
        return []
    if source_id_str.startswith("["):
        try:
            return json.loads(source_id_str)
        except json.JSONDecodeError:
            pass
    from .prompt import GRAPH_FIELD_SEP

    # Fallback to old <SEP> format
    return (
        [s for s in source_id_str.split(GRAPH_FIELD_SEP) if s]
        if GRAPH_FIELD_SEP in source_id_str
        else [source_id_str]
    )
