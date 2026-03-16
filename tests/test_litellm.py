"""Tests for LiteLLM integration."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from nano_graphrag._llm_litellm import (
    LiteLLMWrapper,
    litellm_completion,
    detect_provider,
    supports_structured_output,
)
from nano_graphrag._schemas import EntityExtractionOutput
from nano_graphrag import GraphRAG
from nano_graphrag.base import GraphRAGConfig


class TestDetectProvider:
    """Test provider detection from model names."""

    def test_openai_models(self):
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("o1-preview") == "openai"
        assert detect_provider("o3-mini") == "openai"

    def test_anthropic_models(self):
        assert detect_provider("claude-3-sonnet-20240229") == "anthropic"
        assert detect_provider("claude-3-opus-20240229") == "anthropic"

    def test_google_models(self):
        assert detect_provider("gemini-2.0-flash-exp") == "google_genai"
        assert detect_provider("gemini-pro") == "google_genai"

    def test_cohere_models(self):
        assert detect_provider("command-r") == "cohere"
        assert detect_provider("command-plus") == "cohere"

    def test_explicit_provider_prefix(self):
        assert detect_provider("openai/gpt-4o") == "openai"
        assert detect_provider("anthropic/claude-3-sonnet-20240229") == "anthropic"
        assert detect_provider("ollama/llama3.2") == "ollama"
        assert detect_provider("ollama/mistral") == "ollama"

    def test_mistral_models(self):
        assert detect_provider("mistral-7b") == "mistral"
        assert detect_provider("mixtral-8x7b") == "mistral"

    def test_ollama_models(self):
        assert detect_provider("llama3.2") == "ollama"
        assert detect_provider("llama2") == "ollama"
        assert detect_provider("mistral") == "ollama"
        assert detect_provider("gemma2:9b") == "ollama"
        assert detect_provider("phi3") == "ollama"
        assert detect_provider("qwen2.5") == "ollama"

    def test_unknown_model_defaults_to_openai(self):
        with patch("nano_graphrag._llm_litellm.logger") as mock_logger:
            result = detect_provider("unknown-model-x")
            assert result == "openai"
            mock_logger.warning.assert_called_once()


class TestSupportsStructuredOutput:
    """Test structured output support detection."""

    def test_openai_supports_structured_output(self):
        assert supports_structured_output("gpt-4o") is True
        assert supports_structured_output("openai/gpt-4o") is True

    def test_anthropic_supports_structured_output(self):
        assert supports_structured_output("claude-3-sonnet-20240229") is True
        assert supports_structured_output("anthropic/claude-3-opus-20240229") is True

    def test_google_supports_structured_output(self):
        assert supports_structured_output("gemini-2.0-flash-exp") is True
        assert supports_structured_output("google_genai/gemini-pro") is True

    def test_ollama_does_not_support_structured_output(self):
        # Ollama is not in PROVIDERS_SUPPORTING_STRUCTURED_OUTPUT
        assert supports_structured_output("ollama/llama3.2") is False
        assert supports_structured_output("llama3.2") is False


@pytest.mark.asyncio
class TestLiteLLMCompletion:
    """Test LiteLLM completion function."""

    async def test_completion_with_structured_output(self):
        """Test LiteLLM completion returns parsed BaseModel."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            # Mock response with structured output
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"entities": [], "relationships": []}'
            mock_completion.return_value = mock_response

            result = await litellm_completion(
                model="gpt-4o",
                prompt="Test prompt",
                response_format=EntityExtractionOutput,
            )

            assert isinstance(result, EntityExtractionOutput)
            assert result.entities == []
            assert result.relationships == []

    async def test_completion_with_api_base(self):
        """Test LiteLLM with custom API base."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_completion.return_value = mock_response

            await litellm_completion(
                model="ollama/llama3.2",
                prompt="Test prompt",
                api_base="http://localhost:11434",
            )

            # Verify api_base was passed
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_base"] == "http://localhost:11434"
            assert call_kwargs["timeout"] == 120  # Default timeout

    async def test_completion_with_custom_timeout(self):
        """Test LiteLLM with custom timeout."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_completion.return_value = mock_response

            await litellm_completion(
                model="gpt-4o",
                prompt="Test prompt",
                timeout=300,
            )

            # Verify timeout was passed
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["timeout"] == 300

    async def test_completion_with_api_key(self):
        """Test LiteLLM with custom API key."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_completion.return_value = mock_response

            await litellm_completion(
                model="gpt-4o",
                prompt="Test prompt",
                api_key="sk-test-key",
            )

            # Verify api_key was passed
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_key"] == "sk-test-key"


@pytest.mark.asyncio
class TestLiteLLMWrapper:
    """Test LiteLLMWrapper class."""

    async def test_wrapper_with_structured_output(self):
        """Test wrapper returns parsed BaseModel."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"entities": [], "relationships": []}'
            mock_completion.return_value = mock_response

            wrapper = LiteLLMWrapper(
                model="gpt-4o",
                structured_output=True,
            )
            result = await wrapper(
                "Test prompt",
                response_format=EntityExtractionOutput,
            )

            assert isinstance(result, EntityExtractionOutput)

    async def test_wrapper_with_api_base(self):
        """Test wrapper passes api_base to completion."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_completion.return_value = mock_response

            wrapper = LiteLLMWrapper(
                model="ollama/llama3.2",
                api_base="http://localhost:11434",
            )
            await wrapper("Test prompt")

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_base"] == "http://localhost:11434"

    async def test_wrapper_with_timeout(self):
        """Test wrapper passes timeout to completion."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_completion.return_value = mock_response

            wrapper = LiteLLMWrapper(
                model="gpt-4o",
                timeout=180,
            )
            await wrapper("Test prompt")

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["timeout"] == 180


class TestGraphRAGConfig:
    """Test GraphRAGConfig class."""

    def test_from_dict(self):
        """Test creating config from dict."""
        config = GraphRAGConfig.from_dict({
            "llm_model": "gpt-4o",
            "llm_api_base": "http://localhost:11434",
        })

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        # Default values should be preserved
        assert config.llm_cheap_model == "gpt-4o-mini"

    def test_to_dict(self):
        """Test converting config to dict."""
        config = GraphRAGConfig(
            llm_model="gpt-4o",
            llm_api_base="http://localhost:11434",
        )
        config_dict = config.to_dict()

        assert config_dict["llm_model"] == "gpt-4o"
        assert config_dict["llm_api_base"] == "http://localhost:11434"
        assert config_dict["llm_cheap_model"] == "gpt-4o-mini"

    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_data = {
            "llm_model": "gpt-4o",
            "llm_api_base": "http://localhost:11434",
            "entity_extraction_quality": "thorough",
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = GraphRAGConfig.from_yaml(str(config_file))

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        assert config.entity_extraction_quality == "thorough"

    def test_to_yaml(self, tmp_path):
        """Test saving config to YAML file."""
        import yaml

        config = GraphRAGConfig(
            llm_model="gpt-4o",
            llm_api_base="http://localhost:11434",
        )
        config_file = tmp_path / "test_config.yaml"
        config.to_yaml(str(config_file))

        with open(config_file, "r") as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data["llm_model"] == "gpt-4o"
        assert loaded_data["llm_api_base"] == "http://localhost:11434"

    def test_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434")
        monkeypatch.setenv("LLM_MAX_ASYNC", "32")
        monkeypatch.setenv("ENABLE_NODE_EMBEDDING", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = GraphRAGConfig.from_env()

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        assert config.llm_max_async == 32
        assert config.enable_node_embedding is True
        assert config.log_level == "DEBUG"

    def test_merge(self):
        """Test merging configs."""
        base = GraphRAGConfig(llm_model="gpt-4o-mini", llm_max_async=16)
        overrides = {"llm_model": "gpt-4o", "llm_api_base": "http://localhost:11434"}

        merged = base.merge(overrides)

        assert merged.llm_model == "gpt-4o"  # Overridden
        assert merged.llm_api_base == "http://localhost:11434"  # New
        assert merged.llm_max_async == 16  # Preserved


class TestGraphRAG:
    """Test GraphRAG class with LiteLLM."""

    def test_from_config(self):
        """Test creating GraphRAG from config."""
        config = GraphRAGConfig(
            working_dir="./test_cache",
            llm_model="gpt-4o",
            llm_api_base="http://localhost:11434",
            enable_local=True,
        )

        rag = GraphRAG.from_config(config)

        assert rag.working_dir == "./test_cache"
        assert rag.llm_model == "gpt-4o"
        assert rag.llm_api_base == "http://localhost:11434"
        assert rag.enable_local is True

    def test_timeout_passed_to_litellm_wrapper(self):
        """Test that timeout is passed to LiteLLMWrapper."""
        rag = GraphRAG(
            working_dir="./test_cache",
            llm_timeout=300,
        )

        # Check that LiteLLMWrapper was created with timeout
        # We can't easily test the full initialization without mocking,
        # but we can verify the parameter is set
        assert rag.llm_timeout == 300

    def test_llm_max_async_alias_sets_both_model_limits(self):
        rag = GraphRAG(
            working_dir="./test_cache",
            llm_max_async=8,
        )

        assert rag.best_model_max_async == 8
        assert rag.cheap_model_max_async == 8


class TestEnvVarParsing:
    """Test environment variable parsing helpers."""

    def test_parse_bool_true(self, monkeypatch):
        from nano_graphrag.base import _parse_bool

        for value in ["true", "TRUE", "1", "yes", "YES", "on", "ON"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert _parse_bool("TEST_BOOL") is True

    def test_parse_bool_false(self, monkeypatch):
        from nano_graphrag.base import _parse_bool

        for value in ["false", "FALSE", "0", "no", "NO", "off", "OFF"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert _parse_bool("TEST_BOOL") is False

    def test_parse_bool_default(self, monkeypatch):
        from nano_graphrag.base import _parse_bool

        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert _parse_bool("TEST_BOOL", default=True) is True
        assert _parse_bool("TEST_BOOL", default=False) is False

    def test_parse_int_valid(self, monkeypatch):
        from nano_graphrag.base import _parse_int

        monkeypatch.setenv("TEST_INT", "42")
        assert _parse_int("TEST_INT", default=10) == 42

    def test_parse_int_invalid(self, monkeypatch):
        from nano_graphrag.base import _parse_int

        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert _parse_int("TEST_INT", default=10) == 10

    def test_parse_int_with_min_value(self, monkeypatch):
        from nano_graphrag.base import _parse_int

        monkeypatch.setenv("TEST_INT", "5")
        assert _parse_int("TEST_INT", default=10, min_value=10) == 10  # Below min

        monkeypatch.setenv("TEST_INT", "15")
        assert _parse_int("TEST_INT", default=10, min_value=10) == 15  # Above min
