"""Converter from YuE CLI config to infer.py arguments."""

import subprocess
import tempfile
from pathlib import Path

from yue_cli.models import (
    AudioPromptConfig,
    GenerationConfig,
    OutputConfig,
    RuntimeConfig,
    YueConfig,
)
from yue_cli.parser import ConfigParser


class InferCommand:
    """Represents an infer.py command to be executed."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        genre_text: str,
        lyrics_text: str,
        generation: GenerationConfig,
        audio_prompt: AudioPromptConfig,
        output: OutputConfig,
        runtime: RuntimeConfig,
    ) -> None:
        """Initialize InferCommand with configuration parameters."""
        self.name = name
        self.genre_text = genre_text
        self.lyrics_text = lyrics_text
        self.generation = generation
        self.audio_prompt = audio_prompt
        self.output = output
        self.runtime = runtime

    def build_args(self, genre_path: Path, lyrics_path: Path) -> list[str]:
        """Build command-line arguments for infer.py."""
        args = [
            "--genre_txt",
            str(genre_path),
            "--lyrics_txt",
            str(lyrics_path),
            "--stage1_model",
            self.generation.stage1_model,
            "--stage2_model",
            self.generation.stage2_model,
            "--max_new_tokens",
            str(self.generation.max_new_tokens),
            "--repetition_penalty",
            str(self.generation.repetition_penalty),
            "--run_n_segments",
            str(self.generation.run_n_segments),
            "--stage2_batch_size",
            str(self.generation.stage2_batch_size),
            "--seed",
            str(self.generation.seed),
            "--output_dir",
            self.output.dir,
            "--cuda_idx",
            str(self.runtime.cuda_idx),
        ]

        # Optional flags
        if self.output.keep_intermediate:
            args.append("--keep_intermediate")
        if self.output.rescale:
            args.append("--rescale")
        if self.runtime.disable_offload_model:
            args.append("--disable_offload_model")

        # Audio prompt options
        if self.audio_prompt.mode == "single":
            args.extend(
                [
                    "--use_audio_prompt",
                    "--audio_prompt_path",
                    self.audio_prompt.audio_path,
                    "--prompt_start_time",
                    str(self.audio_prompt.start_time),
                    "--prompt_end_time",
                    str(self.audio_prompt.end_time),
                ]
            )
        elif self.audio_prompt.mode == "dual":
            args.extend(
                [
                    "--use_dual_tracks_prompt",
                    "--vocal_track_prompt_path",
                    self.audio_prompt.vocal_path,
                    "--instrumental_track_prompt_path",
                    self.audio_prompt.instrumental_path,
                    "--prompt_start_time",
                    str(self.audio_prompt.start_time),
                    "--prompt_end_time",
                    str(self.audio_prompt.end_time),
                ]
            )

        return args

    def to_command_string(
        self,
        infer_py_path: str,
        genre_path: Path,
        lyrics_path: Path,
    ) -> str:
        """Generate the full command string for display."""
        args = self.build_args(genre_path, lyrics_path)
        return f"python {infer_py_path} " + " ".join(args)


class ConfigConverter:
    """Converts YueConfig to InferCommand objects."""

    def __init__(self, config: YueConfig) -> None:
        """Initialize ConfigConverter with YueConfig."""
        self.config = config

    def convert(self) -> list[InferCommand]:
        """Convert config to list of InferCommand objects."""
        commands = []
        lyrics_text = self.config.lyrics.to_text()

        variations = self.config.get_effective_variations()
        for name, genre, generation, audio_prompt, output, runtime in variations:
            genre_text = genre.to_text()

            cmd = InferCommand(
                name=name,
                genre_text=genre_text,
                lyrics_text=lyrics_text,
                generation=generation,
                audio_prompt=audio_prompt,
                output=output,
                runtime=runtime,
            )
            commands.append(cmd)

        return commands


class Runner:
    """Executes YuE inference commands."""

    def __init__(
        self,
        yue_path: Path,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize Runner with YuE path and options."""
        self.yue_path = yue_path
        self.infer_py_path = yue_path / "inference" / "infer.py"
        self.dry_run = dry_run

    def run_config(
        self,
        config_path: Path,
        variation_filter: list[str] | None = None,
        output_dir_override: str | None = None,
    ) -> list[dict]:
        """Run inference for a config file."""
        parser = ConfigParser()
        config = parser.parse_file(config_path)

        converter = ConfigConverter(config)
        commands = converter.convert()

        # Filter variations if specified
        if variation_filter:
            commands = [cmd for cmd in commands if cmd.name in variation_filter]

        # Determine base output directory (before any modifications)
        base_output = (
            Path(output_dir_override) if output_dir_override else Path("./output")
        )
        song_name = config.metadata.name

        results = []
        for cmd in commands:
            # Create output subdirectory for this variation
            var_output = base_output / song_name / cmd.name
            cmd.output.dir = str(var_output)

            result = self._run_command(cmd)
            results.append(result)

        return results

    def _run_command(self, cmd: InferCommand) -> dict:
        """Execute a single inference command."""
        result = {
            "name": cmd.name,
            "output_dir": cmd.output.dir,
            "success": False,
            "command": "",
            "error": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            genre_path = tmpdir_path / "genre.txt"
            lyrics_path = tmpdir_path / "lyrics.txt"

            # Write temp files
            genre_path.write_text(cmd.genre_text, encoding="utf-8")
            lyrics_path.write_text(cmd.lyrics_text, encoding="utf-8")

            # Build command
            args = cmd.build_args(genre_path, lyrics_path)
            full_cmd = ["python", str(self.infer_py_path), *args]
            result["command"] = cmd.to_command_string(
                str(self.infer_py_path),
                genre_path,
                lyrics_path,
            )

            if self.dry_run:
                result["success"] = True
                return result

            # Create output directory
            Path(cmd.output.dir).mkdir(parents=True, exist_ok=True)

            # Execute
            try:
                subprocess.run(  # noqa: S603
                    full_cmd,
                    cwd=str(self.yue_path / "inference"),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result["success"] = True
            except subprocess.CalledProcessError as e:
                result["error"] = e.stderr or str(e)
            except OSError as e:
                result["error"] = str(e)

        return result
