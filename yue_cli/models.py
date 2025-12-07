"""Data models for YuE CLI configuration."""

from dataclasses import dataclass, field


@dataclass
class LyricsSection:
    """A section of lyrics (verse, chorus, etc.)."""

    type: str
    text: str


@dataclass
class Lyrics:
    """Lyrics configuration."""

    sections: list[LyricsSection] = field(default_factory=list)
    raw: str | None = None
    file: str | None = None  # External file path (resolved by parser)

    def to_text(self) -> str:
        """Convert lyrics to YuE format."""
        if self.raw:
            return self.raw
        parts = []
        for section in self.sections:
            parts.append(f"[{section.type}]")
            parts.append(section.text.strip())
            parts.append("")
        return "\n".join(parts)


@dataclass
class Genre:
    """Genre configuration."""

    tags: list[str] = field(default_factory=list)
    raw: str | None = None

    def to_text(self) -> str:
        """Convert genre to YuE format (space-separated tags)."""
        if self.raw:
            return self.raw
        return " ".join(self.tags)


@dataclass
class GenerationConfig:
    """Generation parameters."""

    stage1_model: str = "m-a-p/YuE-s1-7B-anneal-en-cot"
    stage2_model: str = "m-a-p/YuE-s2-1B-general"
    max_new_tokens: int = 3000
    repetition_penalty: float = 1.1
    run_n_segments: int = 2
    stage2_batch_size: int = 4
    seed: int = 42


@dataclass
class AudioPromptConfig:
    """Audio prompt configuration for ICL mode."""

    mode: str = "none"  # none | single | dual
    audio_path: str = ""
    vocal_path: str = ""
    instrumental_path: str = ""
    start_time: float = 0.0
    end_time: float = 30.0


@dataclass
class OutputConfig:
    """Output configuration."""

    dir: str = "./output"
    keep_intermediate: bool = False
    rescale: bool = False


@dataclass
class RuntimeConfig:
    """Runtime configuration."""

    cuda_idx: int = 0
    disable_offload_model: bool = False


@dataclass
class Variation:
    """A single variation configuration."""

    name: str
    genre: Genre | None = None
    generation: GenerationConfig | None = None
    audio_prompt: AudioPromptConfig | None = None
    output: OutputConfig | None = None


@dataclass
class Defaults:
    """Default settings for all variations."""

    genre: Genre = field(default_factory=Genre)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    audio_prompt: AudioPromptConfig = field(default_factory=AudioPromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class Metadata:
    """Song metadata."""

    name: str = "untitled"
    description: str = ""


@dataclass
class YueConfig:
    """Complete YuE configuration."""

    version: str = "1.0"
    metadata: Metadata = field(default_factory=Metadata)
    lyrics: Lyrics = field(default_factory=Lyrics)
    # For simple mode (no variations)
    genre: Genre | None = None
    generation: GenerationConfig | None = None
    audio_prompt: AudioPromptConfig | None = None
    output: OutputConfig | None = None
    runtime: RuntimeConfig | None = None
    # For variations mode
    defaults: Defaults | None = None
    variations: list[Variation] = field(default_factory=list)

    def is_variations_mode(self) -> bool:
        """Check if config uses variations mode."""
        return len(self.variations) > 0

    def get_effective_variations(
        self,
    ) -> list[
        tuple[
            str,
            Genre,
            GenerationConfig,
            AudioPromptConfig,
            OutputConfig,
            RuntimeConfig,
        ]
    ]:
        """Get list of effective configurations for each variation."""
        results = []

        if not self.is_variations_mode():
            # Simple mode: single configuration
            genre = self.genre or Genre()
            generation = self.generation or GenerationConfig()
            audio_prompt = self.audio_prompt or AudioPromptConfig()
            output = self.output or OutputConfig()
            runtime = self.runtime or RuntimeConfig()
            results.append(
                (self.metadata.name, genre, generation, audio_prompt, output, runtime),
            )
        else:
            # Variations mode: merge defaults with each variation
            defaults = self.defaults or Defaults()
            for var in self.variations:
                genre = var.genre or defaults.genre
                generation = self._merge_generation(defaults.generation, var.generation)
                audio_prompt = var.audio_prompt or defaults.audio_prompt
                output = var.output or defaults.output
                runtime = defaults.runtime
                results.append(
                    (var.name, genre, generation, audio_prompt, output, runtime),
                )

        return results

    def _merge_generation(
        self,
        default: GenerationConfig,
        override: GenerationConfig | None,
    ) -> GenerationConfig:
        """Merge generation config with defaults."""
        if override is None:
            return default
        return GenerationConfig(
            stage1_model=(
                override.stage1_model
                if override.stage1_model != GenerationConfig().stage1_model
                else default.stage1_model
            ),
            stage2_model=(
                override.stage2_model
                if override.stage2_model != GenerationConfig().stage2_model
                else default.stage2_model
            ),
            max_new_tokens=(
                override.max_new_tokens
                if override.max_new_tokens != GenerationConfig().max_new_tokens
                else default.max_new_tokens
            ),
            repetition_penalty=(
                override.repetition_penalty
                if override.repetition_penalty != GenerationConfig().repetition_penalty
                else default.repetition_penalty
            ),
            run_n_segments=(
                override.run_n_segments
                if override.run_n_segments != GenerationConfig().run_n_segments
                else default.run_n_segments
            ),
            stage2_batch_size=(
                override.stage2_batch_size
                if override.stage2_batch_size != GenerationConfig().stage2_batch_size
                else default.stage2_batch_size
            ),
            seed=(
                override.seed
                if override.seed != GenerationConfig().seed
                else default.seed
            ),
        )
