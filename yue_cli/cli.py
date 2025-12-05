"""CLI interface for YuE."""
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from yue_cli import __version__
from yue_cli.converter import ConfigConverter, Runner
from yue_cli.parser import ConfigParser

app = typer.Typer(
    name="yue",
    help="YAML-based interface for YuE music generation.",
    add_completion=False,
)
console = Console()

# Default YuE path (can be overridden)
DEFAULT_YUE_PATH = Path("/opt/YuE")


def get_yue_path() -> Path:
    """Get YuE installation path."""
    # Check common locations
    candidates = [
        Path("/opt/YuE"),
        Path.cwd() / "YuE",
        Path(__file__).parent.parent / "YuE",
    ]
    for path in candidates:
        if (path / "inference" / "infer.py").exists():
            return path
    return DEFAULT_YUE_PATH


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Path to YAML config file or directory")],
    variation: Annotated[
        Optional[list[str]],
        typer.Option("--variation", "-v", help="Run specific variation(s) only"),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option("--output-dir", "-o", help="Override output directory"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show commands without executing"),
    ] = False,
    yue_path: Annotated[
        Optional[Path],
        typer.Option("--yue-path", help="Path to YuE installation"),
    ] = None,
) -> None:
    """Run YuE inference from a YAML config file or all files in a directory."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Path not found: {config}")
        raise typer.Exit(1)

    yue = yue_path or get_yue_path()

    if config.is_dir():
        yaml_files = sorted(config.glob("*.yaml")) + sorted(config.glob("*.yml"))
        if not yaml_files:
            console.print(f"[yellow]No YAML files found in {config}[/yellow]")
            raise typer.Exit(0)

        console.print(f"[bold]Running directory:[/bold] {config}")
        console.print(f"[bold]Files:[/bold] {len(yaml_files)}")
        if dry_run:
            console.print("[yellow]Dry run mode - commands will not be executed[/yellow]")
        console.print()

        all_results: list[dict] = []
        for yaml_file in yaml_files:
            console.print(f"\n{'─' * 60}")
            console.print(f"[bold cyan]Processing:[/bold cyan] {yaml_file.name}")
            console.print(f"{'─' * 60}")
            results = _run_single_config(yaml_file, variation, output_dir, dry_run, yue)
            all_results.extend(results)

        # Summary table
        console.print(f"\n{'═' * 60}")
        console.print("[bold]Summary[/bold]")
        console.print(f"{'═' * 60}\n")

        table = Table(title="All Results")
        table.add_column("Config", style="cyan")
        table.add_column("Variation", style="cyan")
        table.add_column("Status")
        table.add_column("Output Directory")

        for result in all_results:
            status = "[green]OK[/green]" if result["success"] else "[red]FAILED[/red]"
            table.add_row(
                result.get("config", ""),
                result["name"],
                status,
                result["output_dir"],
            )

        console.print(table)

        # Check for failures
        failures = [r for r in all_results if not r["success"]]
        success_count = len(all_results) - len(failures)
        console.print(f"\n[bold]Total:[/bold] {success_count}/{len(all_results)} succeeded")

        if failures:
            console.print(f"[red]{len(failures)} failed[/red]")
            raise typer.Exit(1)
        else:
            console.print("[green]All variations completed successfully[/green]")
    else:
        results = _run_single_config(config, variation, output_dir, dry_run, yue)
        failures = [r for r in results if not r["success"]]
        if failures:
            raise typer.Exit(1)


def _run_single_config(
    config: Path,
    variation: Optional[list[str]],
    output_dir: Optional[str],
    dry_run: bool,
    yue_path: Path,
) -> list[dict]:
    """Run a single config file. Returns list of results."""
    runner = Runner(yue_path=yue_path, dry_run=dry_run)

    console.print(f"[bold]Running config:[/bold] {config}")

    if dry_run:
        console.print("[yellow]Dry run mode - commands will not be executed[/yellow]")

    results = runner.run_config(
        config_path=config,
        variation_filter=variation,
        output_dir_override=output_dir,
    )

    # Add config name to results
    for r in results:
        r["config"] = config.name

    # Display results
    table = Table(title="Results")
    table.add_column("Variation", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Output Directory")

    for result in results:
        status = "[green]OK[/green]" if result["success"] else "[red]FAILED[/red]"
        table.add_row(result["name"], status, result["output_dir"])

    console.print(table)

    if dry_run:
        console.print("\n[bold]Commands to be executed:[/bold]")
        for result in results:
            console.print(f"\n[cyan]{result['name']}:[/cyan]")
            console.print(f"  {result['command']}")

    # Check for failures
    failures = [r for r in results if not r["success"]]
    if failures:
        for f in failures:
            console.print(f"[red]Error in {f['name']}:[/red] {f.get('error', 'Unknown error')}")

    return results


@app.command(name="list")
def list_variations(
    config: Annotated[Path, typer.Argument(help="Path to YAML config file or directory")],
    commands: Annotated[
        bool,
        typer.Option("--commands", "-c", help="Show infer.py commands for each variation"),
    ] = False,
    lyrics: Annotated[
        bool,
        typer.Option("--lyrics", "-l", help="Show lyrics content"),
    ] = False,
    genre_txt: Annotated[
        str,
        typer.Option("--genre-txt", help="Path placeholder for genre.txt (for command generation)"),
    ] = "{output_dir}/genre.txt",
    lyrics_txt: Annotated[
        str,
        typer.Option("--lyrics-txt", help="Path placeholder for lyrics.txt (for command generation)"),
    ] = "{output_dir}/lyrics.txt",
    yue_path: Annotated[
        Optional[Path],
        typer.Option("--yue-path", help="Path to YuE installation"),
    ] = None,
) -> None:
    """List all variations in a config file or directory."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Path not found: {config}")
        raise typer.Exit(1)

    # If directory, find all YAML files
    if config.is_dir():
        yaml_files = sorted(config.glob("*.yaml")) + sorted(config.glob("*.yml"))
        if not yaml_files:
            console.print(f"[yellow]No YAML files found in {config}[/yellow]")
            raise typer.Exit(0)

        # Count total variations
        total_variations = 0
        file_variations: list[tuple[Path, int]] = []
        parser = ConfigParser()
        for yaml_file in yaml_files:
            cfg = parser.parse_file(yaml_file)
            converter = ConfigConverter(cfg)
            count = len(converter.convert())
            file_variations.append((yaml_file, count))
            total_variations += count

        console.print(f"[bold]Directory:[/bold] {config}")
        console.print(f"[bold]Files:[/bold] {len(yaml_files)}")
        console.print(f"[bold]Total variations:[/bold] {total_variations}")
        console.print()

        for yaml_file, _ in file_variations:
            _list_single_config(yaml_file, commands, lyrics, genre_txt, lyrics_txt, yue_path)
            console.print("\n" + "─" * 60 + "\n")

        # Summary at end
        console.print(f"[bold green]Summary: {len(yaml_files)} files, {total_variations} variations[/bold green]")
    else:
        _list_single_config(config, commands, lyrics, genre_txt, lyrics_txt, yue_path)


def _list_single_config(
    config: Path,
    commands: bool,
    lyrics: bool,
    genre_txt: str,
    lyrics_txt: str,
    yue_path: Optional[Path],
) -> None:
    """List variations for a single config file."""
    parser = ConfigParser()
    cfg = parser.parse_file(config)

    console.print(f"[bold]Config:[/bold] {config}")
    console.print(f"[bold]Song:[/bold] {cfg.metadata.name}")
    console.print(f"[bold]Mode:[/bold] {'Variations' if cfg.is_variations_mode() else 'Simple'}")
    console.print()

    converter = ConfigConverter(cfg)
    cmds = converter.convert()

    # Calculate output directories (without modifying original)
    base_output = Path("./output")
    song_name = cfg.metadata.name
    output_dirs = {}
    for cmd in cmds:
        output_dirs[cmd.name] = str(base_output / song_name / cmd.name)

    # Show lyrics if requested
    if lyrics:
        from rich.text import Text
        console.print("[bold]Lyrics:[/bold]")
        console.print(Text(cfg.lyrics.to_text()))
        console.print()

    # Variations table
    table = Table(title="Variations")
    table.add_column("Name", style="cyan")
    table.add_column("Genre Tags")
    table.add_column("Seed")
    table.add_column("Output Dir")

    for cmd in cmds:
        genre_preview = cmd.genre_text[:40] + "..." if len(cmd.genre_text) > 40 else cmd.genre_text
        table.add_row(
            cmd.name,
            genre_preview,
            str(cmd.generation.seed),
            output_dirs[cmd.name],
        )

    console.print(table)

    # Show commands if requested
    if commands:
        yue = yue_path or get_yue_path()
        infer_py = yue / "inference" / "infer.py"

        console.print("\n[bold]Commands:[/bold]")
        for cmd in cmds:
            out_dir = output_dirs[cmd.name]
            # Replace placeholders
            g_path = genre_txt.replace("{output_dir}", out_dir)
            l_path = lyrics_txt.replace("{output_dir}", out_dir)

            args = _build_command_args(cmd, Path(g_path), Path(l_path), out_dir)
            full_cmd = f"python {infer_py} {' '.join(args)}"

            console.print(f"\n[cyan]{cmd.name}:[/cyan]")
            console.print(full_cmd)

            # Also show genre content
            console.print(f"  [dim]# genre.txt: {cmd.genre_text}[/dim]")


def _build_command_args(cmd, genre_path: Path, lyrics_path: Path, output_dir: str) -> list[str]:
    """Build infer.py command arguments."""
    args = [
        "--genre_txt", str(genre_path),
        "--lyrics_txt", str(lyrics_path),
        "--stage1_model", cmd.generation.stage1_model,
        "--stage2_model", cmd.generation.stage2_model,
        "--max_new_tokens", str(cmd.generation.max_new_tokens),
        "--repetition_penalty", str(cmd.generation.repetition_penalty),
        "--run_n_segments", str(cmd.generation.run_n_segments),
        "--stage2_batch_size", str(cmd.generation.stage2_batch_size),
        "--seed", str(cmd.generation.seed),
        "--output_dir", output_dir,
        "--cuda_idx", str(cmd.runtime.cuda_idx),
    ]

    if cmd.output.keep_intermediate:
        args.append("--keep_intermediate")
    if cmd.output.rescale:
        args.append("--rescale")
    if cmd.runtime.disable_offload_model:
        args.append("--disable_offload_model")

    if cmd.audio_prompt.mode == "single":
        args.extend([
            "--use_audio_prompt",
            "--audio_prompt_path", cmd.audio_prompt.audio_path,
            "--prompt_start_time", str(cmd.audio_prompt.start_time),
            "--prompt_end_time", str(cmd.audio_prompt.end_time),
        ])
    elif cmd.audio_prompt.mode == "dual":
        args.extend([
            "--use_dual_tracks_prompt",
            "--vocal_track_prompt_path", cmd.audio_prompt.vocal_path,
            "--instrumental_track_prompt_path", cmd.audio_prompt.instrumental_path,
            "--prompt_start_time", str(cmd.audio_prompt.start_time),
            "--prompt_end_time", str(cmd.audio_prompt.end_time),
        ])

    return args


@app.command()
def init(
    template: Annotated[
        str,
        typer.Option("--template", "-t", help="Template type: basic or variations"),
    ] = "basic",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("config.yaml"),
) -> None:
    """Create a new config file from a template."""
    templates_dir = Path(__file__).parent / "templates"

    template_map = {
        "basic": "basic.yaml",
        "variations": "variations.yaml",
    }

    if template not in template_map:
        console.print(f"[red]Error:[/red] Unknown template: {template}")
        console.print(f"Available templates: {', '.join(template_map.keys())}")
        raise typer.Exit(1)

    template_file = templates_dir / template_map[template]

    if not template_file.exists():
        console.print(f"[red]Error:[/red] Template file not found: {template_file}")
        raise typer.Exit(1)

    if output.exists():
        if not typer.confirm(f"File {output} already exists. Overwrite?"):
            raise typer.Exit(0)

    shutil.copy(template_file, output)
    console.print(f"[green]Created:[/green] {output}")
    console.print("Edit this file to customize your song configuration.")


@app.command()
def validate(
    config: Annotated[Path, typer.Argument(help="Path to YAML config file or directory")],
) -> None:
    """Validate a config file or all YAML files in a directory."""
    if not config.exists():
        console.print(f"[red]Error:[/red] Path not found: {config}")
        raise typer.Exit(1)

    if config.is_dir():
        yaml_files = sorted(config.glob("*.yaml")) + sorted(config.glob("*.yml"))
        if not yaml_files:
            console.print(f"[yellow]No YAML files found in {config}[/yellow]")
            raise typer.Exit(0)

        console.print(f"[bold]Validating directory:[/bold] {config}")
        console.print(f"[bold]Files:[/bold] {len(yaml_files)}")
        console.print()

        results: list[tuple[Path, bool, str]] = []
        for yaml_file in yaml_files:
            valid, message = _validate_single_config(yaml_file)
            results.append((yaml_file, valid, message))

        # Results table
        table = Table(title="Validation Results")
        table.add_column("File", style="cyan")
        table.add_column("Status")
        table.add_column("Details")

        for path, valid, message in results:
            status = "[green]Valid[/green]" if valid else "[red]Invalid[/red]"
            table.add_row(path.name, status, message)

        console.print(table)

        # Summary
        valid_count = sum(1 for _, v, _ in results if v)
        invalid_count = len(results) - valid_count
        if invalid_count > 0:
            console.print(f"\n[red]{invalid_count} invalid file(s)[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"\n[green]All {valid_count} file(s) valid[/green]")
    else:
        valid, message = _validate_single_config(config)
        if not valid:
            raise typer.Exit(1)


def _validate_single_config(config: Path) -> tuple[bool, str]:
    """Validate a single config file. Returns (is_valid, message)."""
    try:
        parser = ConfigParser()
        cfg = parser.parse_file(config)

        mode = "Variations" if cfg.is_variations_mode() else "Simple"
        variation_count = len(cfg.variations) if cfg.is_variations_mode() else 1

        # Check for warnings
        warnings = []
        if not cfg.lyrics.sections and not cfg.lyrics.raw:
            warnings.append("No lyrics defined")

        message = f"{cfg.metadata.name} ({mode}, {variation_count} var)"
        if warnings:
            message += f" [yellow]⚠ {', '.join(warnings)}[/yellow]"

        console.print(f"[green]Valid:[/green] {config}")
        console.print(f"  Song: {cfg.metadata.name}")
        console.print(f"  Mode: {mode}")
        if cfg.is_variations_mode():
            console.print(f"  Variations: {variation_count}")
        if warnings:
            for w in warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

        return True, message

    except Exception as e:
        console.print(f"[red]Invalid:[/red] {config}")
        console.print(f"  Error: {e}")
        return False, str(e)


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"yue-cli version {__version__}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
