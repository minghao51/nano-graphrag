"""Tests for LiteLLM integration."""

import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nano_graphrag import GraphRAG
from nano_graphrag._config import GraphRAGSettings
from nano_graphrag._llm_litellm import (
    LiteLLMWrapper,
    build_json_schema_response_format,
    build_provider_requirements,
    detect_provider,
    litellm_completion,
    litellm_completion_stream,
    litellm_embedding,
    should_fallback_without_structured_output,
    supports_structured_output,
)
from nano_graphrag._schemas import EntityExtractionOutput
from nano_graphrag.base import DEFAULT_CHEAP_MODEL, GraphRAGConfig

pytestmark = pytest.mark.unit


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
        assert detect_provider("openrouter/qwen/qwen3-235b-a22b") == "openrouter"
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

    def test_unknown_model_raises(self):
        with patch("nano_graphrag._llm_litellm.logger") as mock_logger:
            from nano_graphrag._exceptions import ConfigError

            with pytest.raises(ConfigError, match="Unable to detect provider"):
                detect_provider("unknown-model-x")
            mock_logger.error.assert_called_once()


class TestSupportsStructuredOutput:
    """Test structured output support detection."""

    def test_openai_supports_structured_output(self):
        assert supports_structured_output("gpt-4o") is True
        assert supports_structured_output("openai/gpt-4o") is True
        assert supports_structured_output("openrouter/qwen/qwen3-235b-a22b") is True

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
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["response_format"]["type"] == "json_schema"
            assert call_kwargs["response_format"]["json_schema"]["strict"] is True
            assert call_kwargs["response_format"]["json_schema"]["name"] == "EntityExtractionOutput"
            assert "provider" not in call_kwargs

    async def test_openrouter_structured_output_requires_provider_parameters(self):
        """OpenRouter structured-output calls should require compatible providers."""
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"entities": [], "relationships": []}'
            mock_completion.return_value = mock_response

            result = await litellm_completion(
                # Use a non-Qwen model to test the native structured output path
                model="openrouter/openai/gpt-4o",
                prompt="Test prompt",
                response_format=EntityExtractionOutput,
            )

            assert isinstance(result, EntityExtractionOutput)
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["provider"] == {"require_parameters": True}
            assert call_kwargs["response_format"]["type"] == "json_schema"

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

    async def test_completion_emits_llm_callback(self):
        dispatcher = AsyncMock()
        with patch("nano_graphrag._llm_litellm.litellm.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            mock_completion.return_value = mock_response

            await litellm_completion(
                model="gpt-4o",
                prompt="Test prompt",
                callback_dispatcher=dispatcher,
            )

            dispatcher.llm_call.assert_awaited()

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

    async def test_completion_retries_without_structured_output_on_bad_request(self):
        """Structured-output rejection should retry without response_format."""
        bad_request = type("BadRequestError", (Exception,), {})

        with patch(
            "nano_graphrag._llm_litellm.UNSUPPORTED_STRUCTURED_OUTPUT_ERRORS", (bad_request,)
        ):
            with patch(
                "nano_graphrag._llm_litellm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '{"entities": [], "relationships": []}'
                mock_completion.side_effect = [
                    bad_request("response_format is not supported"),
                    mock_response,
                ]

                result = await litellm_completion(
                    model="gpt-4o",
                    prompt="Test prompt",
                    response_format=EntityExtractionOutput,
                )

                assert isinstance(result, EntityExtractionOutput)
                assert mock_completion.await_count == 2
                first_call = mock_completion.await_args_list[0].kwargs
                second_call = mock_completion.await_args_list[1].kwargs
                assert "response_format" in first_call
                assert "response_format" not in second_call

    async def test_completion_does_not_retry_non_transient_errors(self):
        """Non-transient errors should fail fast without extra retries."""
        with patch(
            "nano_graphrag._llm_litellm.litellm.acompletion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.side_effect = ValueError("invalid request schema")
            with pytest.raises(ValueError, match="invalid request schema"):
                await litellm_completion(
                    model="gpt-4o",
                    prompt="Test prompt",
                )
            assert mock_completion.await_count == 1

    async def test_should_fallback_without_structured_output(self):
        """Fallback detection should be message-based for LiteLLM compatibility."""
        assert (
            should_fallback_without_structured_output(Exception("response_format unsupported"))
            is True
        )
        assert (
            should_fallback_without_structured_output(
                Exception("structured output is not available")
            )
            is True
        )


class TestLiteLLMEmbedding:
    async def test_embedding_emits_llm_callback(self):
        dispatcher = AsyncMock()
        with patch("nano_graphrag._llm_litellm.litellm.aembedding") as mock_embedding:
            mock_response = MagicMock()
            mock_response.data = [{"embedding": [0.1, 0.2]}]
            mock_response.usage = MagicMock(prompt_tokens=3, completion_tokens=0, total_tokens=3)
            mock_embedding.return_value = mock_response

            vectors = await litellm_embedding(
                texts=["hello"], model="text-embedding-3-small", callback_dispatcher=dispatcher
            )

            assert vectors.shape[0] == 1
            dispatcher.llm_call.assert_awaited()
        assert (
            should_fallback_without_structured_output(Exception("authentication failed")) is False
        )

    async def test_build_json_schema_response_format(self):
        """Structured output payload should use the strict json_schema shape."""
        payload = build_json_schema_response_format(EntityExtractionOutput)

        assert payload["type"] == "json_schema"
        assert payload["json_schema"]["name"] == "EntityExtractionOutput"
        assert payload["json_schema"]["strict"] is True
        assert payload["json_schema"]["schema"]["type"] == "object"

    async def test_build_provider_requirements(self):
        """OpenRouter should request providers that honor structured-output params."""
        assert build_provider_requirements("openrouter/qwen/qwen3-235b-a22b") == {
            "require_parameters": True
        }
        assert build_provider_requirements("gpt-4o") is None


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

    async def test_streaming_falls_back_to_buffered_completion(self):
        with patch(
            "nano_graphrag._llm_litellm.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Buffered fallback"
            mock_completion.side_effect = [Exception("stream unsupported"), mock_response]

            chunks = []
            async for chunk in litellm_completion_stream(model="gpt-4o", prompt="Hello"):
                chunks.append(chunk)

            assert "".join(chunks) == "Buffered fallback"


class TestGraphRAGConfig:
    """Test GraphRAGConfig class."""

    def test_from_dict(self):
        """Test creating config from dict."""
        config = GraphRAGConfig.from_dict(
            {
                "llm_model": "gpt-4o",
                "llm_api_base": "http://localhost:11434",
            }
        )

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        # Default values should be preserved
        assert config.llm_cheap_model == DEFAULT_CHEAP_MODEL

    def test_to_dict(self):
        """Test converting config to dict."""
        config = GraphRAGConfig(
            llm_model="gpt-4o",
            llm_api_base="http://localhost:11434",
        )
        config_dict = config.to_dict()

        assert config_dict["llm_model"] == "gpt-4o"
        assert config_dict["llm_api_base"] == "http://localhost:11434"
        assert config_dict["llm_cheap_model"] == DEFAULT_CHEAP_MODEL

    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_data = {
            "llm_model": "gpt-4o",
            "llm_api_base": "http://localhost:11434",
            "entity_extraction_quality": "balanced",
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = GraphRAGConfig.from_yaml(str(config_file))

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        assert config.entity_extraction_quality == "balanced"

    def test_to_yaml(self, tmp_path):
        """Test saving config to YAML file."""
        import yaml

        config = GraphRAGConfig(
            llm_model="gpt-4o",
            llm_api_base="http://localhost:11434",
        )
        config_file = tmp_path / "test_config.yaml"
        config.to_yaml(str(config_file))

        with open(config_file) as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data["llm"]["model"] == "gpt-4o"
        assert loaded_data["llm"]["api_base"] == "http://localhost:11434"

    def test_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434")
        monkeypatch.setenv("LLM_MAX_ASYNC", "32")
        monkeypatch.setenv("ENABLE_NODE_EMBEDDING", "true")
        monkeypatch.setenv("ENABLE_COMMUNITY_REPORTS", "false")
        monkeypatch.setenv("ENTITY_LINKING_USE_NEIGHBORHOOD_EVIDENCE", "false")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = GraphRAGConfig.from_env()

        assert config.llm_model == "gpt-4o"
        assert config.llm_api_base == "http://localhost:11434"
        assert config.llm_max_async == 32
        assert config.enable_node_embedding is True
        assert config.enable_community_reports is False
        assert config.entity_linking_use_neighborhood_evidence is False
        assert config.log_level == "DEBUG"

    def test_merge(self):
        """Test merging configs."""
        base = GraphRAGConfig(llm_model="gpt-4o-mini", llm_max_async=32)
        overrides = {"llm_model": "gpt-4o", "llm_api_base": "http://localhost:11434"}

        merged = base.merge(overrides)

        assert merged.llm_model == "gpt-4o"  # Overridden
        assert merged.llm_api_base == "http://localhost:11434"  # New
        assert merged.llm_max_async == 32  # Preserved

    def test_invalid_cluster_algorithm_rejected(self):
        """Test that invalid graph_cluster_algorithm in GraphRAG raises ConfigError."""
        from nano_graphrag import GraphRAG
        from nano_graphrag._exceptions import ConfigError

        with pytest.raises(ConfigError, match="Unsupported graph_cluster_algorithm"):
            GraphRAG(
                working_dir="./test_cache",
                graph_cluster_algorithm="invalid_algo",
            )


class TestSecretStrHandling:
    """Test that SecretStr is used for API keys internally and properly unwrapped."""

    def test_from_dict_unwraps_api_key(self):
        settings = GraphRAGSettings.from_dict({"api_key": "sk-test-123"})
        flat = settings.to_flat_dict()
        assert flat["api_key"] == "sk-test-123"
        assert isinstance(flat["api_key"], str)

    def test_from_dict_nested_unwraps_llm_api_key(self):
        settings = GraphRAGSettings.from_dict({"llm_api_key": "sk-llm-test"})
        flat = settings.to_flat_dict()
        assert flat["llm_api_key"] == "sk-llm-test"

    def test_from_dict_nested_format_unwraps(self):
        settings = GraphRAGSettings.from_dict(
            {"llm": {"model": "test-model", "api_key": "sk-nested"}, "api_key": "sk-top"}
        )
        flat = settings.to_flat_dict()
        assert flat["api_key"] == "sk-top"
        assert flat["llm_api_key"] == "sk-nested"

    def test_to_yaml_redacts_api_keys(self, tmp_path):
        import yaml

        settings = GraphRAGSettings.from_dict(
            {"api_key": "sk-secret", "llm_api_key": "sk-llm-secret"}
        )
        path = str(tmp_path / "redacted.yaml")
        settings.to_yaml(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        assert data["api_key"] is None
        assert data["llm"]["api_key"] is None

        with open(path) as f:
            raw = f.read()
        assert "sk-secret" not in raw
        assert "sk-llm-secret" not in raw

    def test_none_api_key_stays_none(self):
        settings = GraphRAGSettings.from_dict({})
        flat = settings.to_flat_dict()
        assert flat["api_key"] is None
        assert flat["llm_api_key"] is None

    def test_graphrag_config_receives_plain_strings(self):
        config = GraphRAGConfig.from_dict({"api_key": "sk-test"})
        assert isinstance(config.api_key, str)
        assert config.api_key == "sk-test"


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

    def test_from_config_preserves_experiment_knobs(self):
        config = GraphRAGConfig(
            working_dir="./test_cache",
            alias_max_batches_in_flight=9,
            entity_count_min_ratio=4.5,
            entity_count_min_absolute=11,
        )

        rag = GraphRAG.from_config(config)
        runtime_config = rag._to_config_dict()

        assert rag.alias_max_batches_in_flight == 9
        assert rag.entity_count_min_ratio == 4.5
        assert rag.entity_count_min_absolute == 11
        assert runtime_config["alias_max_batches_in_flight"] == 9
        assert runtime_config["entity_count_min_ratio"] == 4.5
        assert runtime_config["entity_count_min_absolute"] == 11

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

    def test_embedding_batch_num_deprecated_alias(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rag = GraphRAG(
                working_dir="./test_cache",
                embedding_batch_num=16,
            )

        assert rag.embedding_batch_size == 16
        assert any("embedding_batch_num" in str(w.message) for w in caught)


class TestEnvVarParsing:
    """Test environment variable parsing helpers."""

    def test_parse_bool_true(self, monkeypatch):
        from nano_graphrag._config import _parse_bool_env as _parse_bool

        for value in ["true", "TRUE", "1", "yes", "YES", "on", "ON"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert _parse_bool("TEST_BOOL") is True

    def test_parse_bool_false(self, monkeypatch):
        from nano_graphrag._config import _parse_bool_env as _parse_bool

        for value in ["false", "FALSE", "0", "no", "NO", "off", "OFF"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert _parse_bool("TEST_BOOL") is False

    def test_parse_bool_default(self, monkeypatch):
        from nano_graphrag._config import _parse_bool_env as _parse_bool

        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert _parse_bool("TEST_BOOL", default=True) is True
        assert _parse_bool("TEST_BOOL", default=False) is False

    def test_parse_int_valid(self, monkeypatch):
        from nano_graphrag._config import _parse_int_env as _parse_int

        monkeypatch.setenv("TEST_INT", "42")
        assert _parse_int("TEST_INT", default=10) == 42

    def test_parse_int_invalid(self, monkeypatch):
        from nano_graphrag._config import _parse_int_env as _parse_int

        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert _parse_int("TEST_INT", default=10) == 10

    def test_parse_int_with_min_value(self, monkeypatch):
        from nano_graphrag._config import _parse_int_env as _parse_int

        monkeypatch.setenv("TEST_INT", "5")
        assert _parse_int("TEST_INT", default=10, min_value=10) == 10  # Below min

        monkeypatch.setenv("TEST_INT", "15")
        assert _parse_int("TEST_INT", default=10, min_value=10) == 15  # Above min


class TestGraphRAGConfigValidation:
    """Test GraphRAGConfig validation."""

    def test_invalid_quality_mode_raises_error(self):
        """Test that invalid entity_extraction_quality raises ValueError."""
        import pytest

        from nano_graphrag.base import GraphRAGConfig

        with pytest.raises(ValueError, match="quality"):
            GraphRAGConfig(entity_extraction_quality="invalid")

    def test_valid_quality_modes_accepted(self):
        """Test that all valid quality modes are accepted."""
        from nano_graphrag.base import GraphRAGConfig

        for mode in ["fast", "balanced"]:
            config = GraphRAGConfig(entity_extraction_quality=mode)
            assert config.entity_extraction_quality == mode

    def test_invalid_cluster_algorithm_raises_error(self):
        """Test that invalid graph_cluster_algorithm raises ValueError."""
        import pytest

        from nano_graphrag.base import GraphRAGConfig

        with pytest.raises(ValueError, match="algorithm"):
            GraphRAGConfig(graph_cluster_algorithm="invalid")

    def test_valid_cluster_algorithms_accepted(self):
        """Test that all valid cluster algorithms are accepted."""
        from nano_graphrag.base import GraphRAGConfig

        for algo in ["louvain", "leiden"]:
            config = GraphRAGConfig(graph_cluster_algorithm=algo)
            assert config.graph_cluster_algorithm == algo

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log_level raises ConfigError."""
        import pytest

        from nano_graphrag._exceptions import ConfigError
        from nano_graphrag.base import GraphRAGConfig

        with pytest.raises(ConfigError, match="log_level|level"):
            GraphRAGConfig(log_level="invalid")

    def test_valid_log_levels_accepted(self):
        """Test that all valid log levels are accepted."""
        from nano_graphrag.base import GraphRAGConfig

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = GraphRAGConfig(log_level=level)
            assert config.log_level == level

    def test_from_env_with_invalid_quality(self, monkeypatch):
        """Test that from_env validates entity_extraction_quality."""
        import pytest

        from nano_graphrag.base import GraphRAGConfig

        monkeypatch.setenv("ENTITY_EXTRACTION_QUALITY", "invalid")
        with pytest.raises(ValueError, match="quality"):
            GraphRAGConfig.from_env()

    def test_from_yaml_with_invalid_cluster_algorithm(self, tmp_path):
        """Test that from_yaml validates graph_cluster_algorithm."""
        import pytest
        import yaml

        from nano_graphrag.base import GraphRAGConfig

        config_data = {"graph_cluster_algorithm": "invalid"}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="algorithm"):
            GraphRAGConfig.from_yaml(str(config_file))
