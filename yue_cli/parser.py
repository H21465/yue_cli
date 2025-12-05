"""YAML parser for YuE CLI configuration."""

from pathlib import Path
from typing import Any

import yaml

from yue_cli.models import (
    AudioPromptConfig,
    Defaults,
    GenerationConfig,
    Genre,
    Lyrics,
    LyricsSection,
    Metadata,
    OutputConfig,
    RuntimeConfig,
    Variation,
    YueConfig,
)


class ConfigParser:
    """Parser for YuE YAML configuration files."""

    def __init__(self) -> None:
        """Initialize parser."""
        self._config_dir: Path | None = None

    def parse_file(self, path: Path | str) -> YueConfig:
        """Parse a YAML configuration file."""
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        # Store config directory for resolving relative paths
        self._config_dir = path.parent

        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self.parse_dict(data)

    def parse_dict(self, data: dict[str, Any]) -> YueConfig:
        """Parse a dictionary into YueConfig."""
        config = YueConfig()

        # Version
        config.version = data.get("version", "1.0")

        # Metadata
        if "metadata" in data:
            config.metadata = self._parse_metadata(data["metadata"])

        # Lyrics (required)
        if "lyrics" in data:
            config.lyrics = self._parse_lyrics(data["lyrics"])

        # Check mode: variations or simple
        if "variations" in data:
            # Variations mode
            if "defaults" in data:
                config.defaults = self._parse_defaults(data["defaults"])
            config.variations = [self._parse_variation(v) for v in data["variations"]]
        else:
            # Simple mode
            if "genre" in data:
                config.genre = self._parse_genre(data["genre"])
            if "generation" in data:
                config.generation = self._parse_generation(data["generation"])
            if "audio_prompt" in data:
                config.audio_prompt = self._parse_audio_prompt(data["audio_prompt"])
            if "output" in data:
                config.output = self._parse_output(data["output"])
            if "runtime" in data:
                config.runtime = self._parse_runtime(data["runtime"])

        return config

    def _parse_metadata(self, data: dict[str, Any]) -> Metadata:
        """Parse metadata section."""
        return Metadata(
            name=data.get("name", "untitled"),
            description=data.get("description", ""),
        )

    def _parse_lyrics(self, data: dict[str, Any]) -> Lyrics:
        """Parse lyrics section."""
        lyrics = Lyrics()

        if "file" in data:
            # External file reference
            file_path = data["file"]
            lyrics.file = file_path

            # Resolve relative path from config directory
            if self._config_dir is not None:
                resolved_path = self._config_dir / file_path
            else:
                resolved_path = Path(file_path)

            if not resolved_path.exists():
                msg = f"Lyrics file not found: {resolved_path}"
                raise FileNotFoundError(msg)

            # Read file content as raw lyrics
            lyrics.raw = resolved_path.read_text(encoding="utf-8")
        elif "raw" in data:
            lyrics.raw = data["raw"]
        elif "sections" in data:
            lyrics.sections = [
                LyricsSection(
                    type=s.get("type", "verse"),
                    text=s.get("text", ""),
                )
                for s in data["sections"]
            ]

        return lyrics

    def _parse_genre(self, data: dict[str, Any]) -> Genre:
        """Parse genre section."""
        genre = Genre()

        if "raw" in data:
            genre.raw = data["raw"]
        elif "tags" in data:
            genre.tags = data["tags"]

        return genre

    def _parse_generation(self, data: dict[str, Any]) -> GenerationConfig:
        """Parse generation section."""
        return GenerationConfig(
            stage1_model=data.get("stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot"),
            stage2_model=data.get("stage2_model", "m-a-p/YuE-s2-1B-general"),
            max_new_tokens=data.get("max_new_tokens", 3000),
            repetition_penalty=data.get("repetition_penalty", 1.1),
            run_n_segments=data.get("run_n_segments", 2),
            stage2_batch_size=data.get("stage2_batch_size", 4),
            seed=data.get("seed", 42),
        )

    def _parse_audio_prompt(self, data: dict[str, Any]) -> AudioPromptConfig:
        """Parse audio_prompt section."""
        return AudioPromptConfig(
            mode=data.get("mode", "none"),
            audio_path=data.get("audio_path", ""),
            vocal_path=data.get("vocal_path", ""),
            instrumental_path=data.get("instrumental_path", ""),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 30.0),
        )

    def _parse_output(self, data: dict[str, Any]) -> OutputConfig:
        """Parse output section."""
        return OutputConfig(
            dir=data.get("dir", "./output"),
            keep_intermediate=data.get("keep_intermediate", False),
            rescale=data.get("rescale", False),
        )

    def _parse_runtime(self, data: dict[str, Any]) -> RuntimeConfig:
        """Parse runtime section."""
        return RuntimeConfig(
            cuda_idx=data.get("cuda_idx", 0),
            disable_offload_model=data.get("disable_offload_model", False),
        )

    def _parse_defaults(self, data: dict[str, Any]) -> Defaults:
        """Parse defaults section."""
        defaults = Defaults()

        if "generation" in data:
            defaults.generation = self._parse_generation(data["generation"])
        if "audio_prompt" in data:
            defaults.audio_prompt = self._parse_audio_prompt(data["audio_prompt"])
        if "output" in data:
            defaults.output = self._parse_output(data["output"])
        if "runtime" in data:
            defaults.runtime = self._parse_runtime(data["runtime"])

        return defaults

    def _parse_variation(self, data: dict[str, Any]) -> Variation:
        """Parse a single variation."""
        variation = Variation(name=data.get("name", "unnamed"))

        if "genre" in data:
            variation.genre = self._parse_genre(data["genre"])
        if "generation" in data:
            variation.generation = self._parse_generation(data["generation"])
        if "audio_prompt" in data:
            variation.audio_prompt = self._parse_audio_prompt(data["audio_prompt"])
        if "output" in data:
            variation.output = self._parse_output(data["output"])

        return variation
