from __future__ import annotations

import pytest

try:
    from typer.testing import CliRunner

    from nano_graphrag._cli import app

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False

pytestmark = pytest.mark.skipif(not HAS_TYPER, reason="typer not installed")


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def cli_working_dir(tmp_path):
    return str(tmp_path / "cli_test_workdir")


class TestCLIHelp:
    def test_main_help(self, cli_runner):
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Nano-GraphRAG" in result.output

    def test_query_help(self, cli_runner):
        result = cli_runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output
        assert "--verbose" in result.output

    def test_insert_help(self, cli_runner):
        result = cli_runner.invoke(app, ["insert", "--help"])
        assert result.exit_code == 0
        assert "--recursive" in result.output

    def test_status_help(self, cli_runner):
        result = cli_runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--working-dir" in result.output

    def test_config_help(self, cli_runner):
        result = cli_runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_refine_help(self, cli_runner):
        result = cli_runner.invoke(app, ["refine", "--help"])
        assert result.exit_code == 0
        assert "--phases" in result.output

    def test_rebuild_help(self, cli_runner):
        result = cli_runner.invoke(app, ["rebuild", "--help"])
        assert result.exit_code == 0

    def test_export_help(self, cli_runner):
        result = cli_runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output


class TestCLIStatus:
    def test_status_nonexistent_dir(self, cli_runner):
        result = cli_runner.invoke(app, ["status", "--working-dir", "/nonexistent/path/xyz"])
        assert result.exit_code == 1


class TestCLIInsert:
    def test_insert_nonexistent_path(self, cli_runner, cli_working_dir):
        result = cli_runner.invoke(
            app, ["insert", "/nonexistent/file.txt", "--working-dir", cli_working_dir]
        )
        assert result.exit_code == 1

    def test_insert_file(self, cli_runner, cli_working_dir, tmp_path):
        """Test insert command parses the file and starts the pipeline (may fail on API auth)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world this is a test document.")
        result = cli_runner.invoke(
            app, ["insert", str(test_file), "--working-dir", cli_working_dir]
        )
        assert (
            result.exit_code == 0
            or "Authentication" in result.output
            or "auth" in str(result.exception).lower()
        )

    def test_insert_empty_dir(self, cli_runner, cli_working_dir, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        result = cli_runner.invoke(
            app, ["insert", str(empty_dir), "--working-dir", cli_working_dir]
        )
        assert result.exit_code == 1


class TestCLIConfig:
    def test_config_init(self, cli_runner, cli_working_dir):
        result = cli_runner.invoke(app, ["config", "init", "--working-dir", cli_working_dir])
        assert "Config written" in result.output

    def test_config_show(self, cli_runner, cli_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=cli_working_dir)
        result = cli_runner.invoke(app, ["config", "show", "--working-dir", cli_working_dir])
        assert result.exit_code == 0
