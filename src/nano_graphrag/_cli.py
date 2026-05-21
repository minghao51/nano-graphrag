from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

try:
    import typer
except ImportError:

    def _cli_unavailable(*args, **kwargs):
        print(
            "CLI requires the [cli] extra. Install with: pip install nano-graphrag[cli]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    app = _cli_unavailable
else:
    app = typer.Typer(
        name="nano-graphrag",
        help="Nano-GraphRAG — lightweight knowledge graph RAG toolkit.",
        no_args_is_help=True,
    )

    @app.command()
    def query(
        question: Annotated[str, typer.Argument(help="The question to ask")],
        mode: Annotated[str, typer.Option("--mode", "-m", help="Query mode")] = "global",
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
        stream: Annotated[bool, typer.Option("--stream", help="Stream response")] = False,
        verbose: Annotated[bool, typer.Option("--verbose", help="Show stream metadata")] = False,
    ):
        """Query the knowledge graph."""
        valid_modes = {"local", "global", "naive", "entity_grounded"}
        if mode not in valid_modes:
            typer.echo(
                f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid_modes))}", err=True
            )
            raise typer.Exit(1)

        from nano_graphrag import GraphRAG, QueryParam

        rag = GraphRAG(working_dir=working_dir)
        param = QueryParam(mode=mode)

        if stream:

            async def _stream():
                from nano_graphrag import StreamComplete, StreamTextChunk

                async for chunk in rag.astream_query(question, param):
                    if isinstance(chunk, StreamTextChunk):
                        print(chunk.text, end="", flush=True)
                    elif verbose and isinstance(chunk, StreamComplete):
                        typer.echo(f"\n[stream complete] latency_ms={chunk.latency_ms}")
                print()

            asyncio.run(_stream())
        else:
            result = rag.query(question, param)
            print(result.answer)

    @app.command()
    def insert(
        path: Annotated[str, typer.Argument(help="File or directory to insert")],
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
        recursive: Annotated[
            bool, typer.Option("--recursive", "-r", help="Recurse into directories")
        ] = False,
    ):
        """Insert documents into the knowledge graph."""
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=working_dir)
        target = Path(path)

        if target.is_file():
            text = target.read_text()
            rag.insert(text)
            typer.echo(f"Inserted: {target}")
        elif target.is_dir():
            patterns = ["*.txt", "*.md"]
            files = []
            for pattern in patterns:
                if recursive:
                    files.extend(target.rglob(pattern))
                else:
                    files.extend(target.glob(pattern))
            if not files:
                typer.echo(f"No .txt/.md files found in {target}")
                raise typer.Exit(1)
            for f in sorted(files):
                text = f.read_text()
                rag.insert(text)
                typer.echo(f"Inserted: {f}")
            typer.echo(f"Total: {len(files)} files")
        else:
            typer.echo(f"Path not found: {path}", err=True)
            raise typer.Exit(1)

    @app.command()
    def status(
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
    ):
        """Show graph status: entity count, communities, storage sizes."""
        if not os.path.exists(working_dir):
            typer.echo(f"Working directory not found: {working_dir}")
            raise typer.Exit(1)

        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=working_dir)
        graph_status = rag.status()

        stats = {
            "documents": graph_status.document_count,
            "chunks": graph_status.chunk_count,
            "communities": graph_status.community_count,
            "entities": graph_status.entity_count,
            "relationships": graph_status.relationship_count,
            "health": graph_status.health,
        }

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Nano-GraphRAG Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            if graph_status.entity_types:
                table.add_row("─── Entity Types ───", "")
                for etype, count in sorted(graph_status.entity_types.items()):
                    table.add_row(f"  {etype}", str(count))
            if graph_status.warnings:
                table.add_row("─── Warnings ───", "")
                for w in graph_status.warnings:
                    table.add_row(f"  ⚠ {w}", "")
            console.print(table)
        except ImportError:
            for key, value in stats.items():
                typer.echo(f"  {key}: {value}")
            if graph_status.warnings:
                typer.echo("  warnings:")
                for w in graph_status.warnings:
                    typer.echo(f"    - {w}")

    @app.command()
    def refine(
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
        phases: Annotated[
            str | None,
            typer.Option("--phases", "-p", help="Comma-separated phases: merge,enrich,infer"),
        ] = None,
    ):
        """Run the knowledge graph refinement pipeline."""
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=working_dir)
        phase_list = phases.split(",") if phases else None
        result = rag.refine(phases=phase_list)
        typer.echo(f"Refinement complete: {json.dumps(result, default=str)}")

    @app.command()
    def rebuild(
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
    ):
        """Rebuild the knowledge graph from existing manifests."""
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=working_dir)
        rag.rebuild_graph()
        typer.echo("Graph rebuild complete.")

    @app.command()
    def export(
        output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "./vault",
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
        include_communities: Annotated[
            bool, typer.Option("--communities", help="Include community reports")
        ] = True,
    ):
        """Export the knowledge graph to an Obsidian vault."""
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=working_dir)
        rag.export_vault(path=output, include_communities=include_communities)
        typer.echo(f"Vault exported to: {output}")

    @app.command()
    def config(
        action: Annotated[str, typer.Argument(help="Action: show or init")] = "show",
        working_dir: Annotated[
            str, typer.Option("--working-dir", "-w", help="Working directory")
        ] = "./nano_graphrag",
    ):
        """Show or initialize configuration."""
        if action == "init":
            from nano_graphrag import GraphRAGConfig

            config_path = os.path.join(working_dir, "settings.yaml")
            if os.path.exists(config_path):
                typer.echo(
                    f"Config already exists: {config_path}. Remove it first to reinitialize.",
                    err=True,
                )
                raise typer.Exit(1)
            config = GraphRAGConfig(working_dir=working_dir)
            os.makedirs(working_dir, exist_ok=True)
            config.to_yaml(config_path)
            typer.echo(f"Config written to: {config_path}")
        elif action == "show":
            from nano_graphrag import GraphRAG

            rag = GraphRAG(working_dir=working_dir)
            safe_config = rag._to_safe_log_dict()
            try:
                from rich.console import Console
                from rich.panel import Panel

                console = Console()
                console.print(
                    Panel(json.dumps(safe_config, indent=2, default=str), title="Configuration")
                )
            except ImportError:
                typer.echo(json.dumps(safe_config, indent=2, default=str))
        else:
            typer.echo(f"Unknown action: {action}. Use 'show' or 'init'.", err=True)
            raise typer.Exit(1)
