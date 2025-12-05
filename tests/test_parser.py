"""Tests for YuE CLI parser."""
import pytest
from pathlib import Path

from yue_cli.parser import ConfigParser
from yue_cli.models import YueConfig


class TestConfigParser:
    """Test cases for ConfigParser."""

    @pytest.fixture
    def parser(self) -> ConfigParser:
        """Create a parser instance."""
        return ConfigParser()

    def test_parse_simple_config(self, parser: ConfigParser) -> None:
        """Test parsing a simple configuration without variations."""
        data = {
            "version": "1.0",
            "metadata": {
                "name": "test-song",
                "description": "A test song",
            },
            "lyrics": {
                "sections": [
                    {"type": "verse", "text": "First verse line"},
                    {"type": "chorus", "text": "Chorus line"},
                ]
            },
            "genre": {
                "tags": ["pop", "female", "uplifting"],
            },
            "generation": {
                "seed": 123,
            },
        }

        config = parser.parse_dict(data)

        assert config.version == "1.0"
        assert config.metadata.name == "test-song"
        assert config.metadata.description == "A test song"
        assert len(config.lyrics.sections) == 2
        assert config.lyrics.sections[0].type == "verse"
        assert config.lyrics.sections[0].text == "First verse line"
        assert config.genre is not None
        assert config.genre.tags == ["pop", "female", "uplifting"]
        assert config.generation is not None
        assert config.generation.seed == 123
        assert not config.is_variations_mode()

    def test_parse_variations_config(self, parser: ConfigParser) -> None:
        """Test parsing a configuration with variations."""
        data = {
            "version": "1.0",
            "metadata": {
                "name": "my-song",
            },
            "lyrics": {
                "sections": [
                    {"type": "verse", "text": "Verse lyrics"},
                ]
            },
            "defaults": {
                "generation": {
                    "stage1_model": "m-a-p/YuE-s1-7B-anneal-en-cot",
                    "seed": 42,
                },
                "output": {
                    "dir": "./output",
                },
            },
            "variations": [
                {
                    "name": "pop-v1",
                    "genre": {"tags": ["pop", "female"]},
                    "generation": {"seed": 100},
                },
                {
                    "name": "rock-v1",
                    "genre": {"tags": ["rock", "male"]},
                    "generation": {"seed": 200},
                },
            ],
        }

        config = parser.parse_dict(data)

        assert config.is_variations_mode()
        assert len(config.variations) == 2
        assert config.variations[0].name == "pop-v1"
        assert config.variations[0].genre.tags == ["pop", "female"]
        assert config.variations[0].generation.seed == 100
        assert config.variations[1].name == "rock-v1"
        assert config.variations[1].genre.tags == ["rock", "male"]
        assert config.variations[1].generation.seed == 200

    def test_lyrics_to_text_sections(self, parser: ConfigParser) -> None:
        """Test converting lyrics sections to YuE format."""
        data = {
            "lyrics": {
                "sections": [
                    {"type": "verse", "text": "Line 1\nLine 2"},
                    {"type": "chorus", "text": "Chorus line"},
                ]
            },
        }

        config = parser.parse_dict(data)
        text = config.lyrics.to_text()

        assert "[verse]" in text
        assert "Line 1" in text
        assert "Line 2" in text
        assert "[chorus]" in text
        assert "Chorus line" in text

    def test_lyrics_to_text_raw(self, parser: ConfigParser) -> None:
        """Test converting raw lyrics to YuE format."""
        raw_lyrics = "[verse]\nRaw lyrics here"
        data = {
            "lyrics": {
                "raw": raw_lyrics,
            },
        }

        config = parser.parse_dict(data)
        text = config.lyrics.to_text()

        assert text == raw_lyrics

    def test_genre_to_text_tags(self, parser: ConfigParser) -> None:
        """Test converting genre tags to YuE format."""
        data = {
            "lyrics": {"sections": []},
            "genre": {
                "tags": ["pop", "female", "uplifting"],
            },
        }

        config = parser.parse_dict(data)
        text = config.genre.to_text()

        assert text == "pop female uplifting"

    def test_genre_to_text_raw(self, parser: ConfigParser) -> None:
        """Test converting raw genre to YuE format."""
        raw_genre = "inspiring female electronic"
        data = {
            "lyrics": {"sections": []},
            "genre": {
                "raw": raw_genre,
            },
        }

        config = parser.parse_dict(data)
        text = config.genre.to_text()

        assert text == raw_genre

    def test_get_effective_variations_simple_mode(self, parser: ConfigParser) -> None:
        """Test getting effective variations in simple mode."""
        data = {
            "metadata": {"name": "simple-song"},
            "lyrics": {"sections": [{"type": "verse", "text": "Lyrics"}]},
            "genre": {"tags": ["pop"]},
            "generation": {"seed": 42},
        }

        config = parser.parse_dict(data)
        variations = config.get_effective_variations()

        assert len(variations) == 1
        name, genre, generation, audio_prompt, output, runtime = variations[0]
        assert name == "simple-song"
        assert genre.tags == ["pop"]
        assert generation.seed == 42

    def test_get_effective_variations_with_defaults(self, parser: ConfigParser) -> None:
        """Test getting effective variations with defaults merged."""
        data = {
            "metadata": {"name": "multi-song"},
            "lyrics": {"sections": [{"type": "verse", "text": "Lyrics"}]},
            "defaults": {
                "generation": {
                    "max_new_tokens": 5000,
                    "seed": 42,
                },
            },
            "variations": [
                {
                    "name": "var1",
                    "genre": {"tags": ["pop"]},
                    "generation": {"seed": 100},  # Override seed
                },
                {
                    "name": "var2",
                    "genre": {"tags": ["rock"]},
                    # Use default seed
                },
            ],
        }

        config = parser.parse_dict(data)
        variations = config.get_effective_variations()

        assert len(variations) == 2

        # First variation: seed overridden
        name1, genre1, gen1, _, _, _ = variations[0]
        assert name1 == "var1"
        assert gen1.seed == 100
        assert gen1.max_new_tokens == 5000  # From defaults

        # Second variation: default seed
        name2, genre2, gen2, _, _, _ = variations[1]
        assert name2 == "var2"
        assert gen2.seed == 42  # Default
        assert gen2.max_new_tokens == 5000  # From defaults

    def test_parse_audio_prompt_single(self, parser: ConfigParser) -> None:
        """Test parsing audio prompt in single mode."""
        data = {
            "lyrics": {"sections": []},
            "audio_prompt": {
                "mode": "single",
                "audio_path": "/path/to/audio.mp3",
                "start_time": 5.0,
                "end_time": 20.0,
            },
        }

        config = parser.parse_dict(data)

        assert config.audio_prompt.mode == "single"
        assert config.audio_prompt.audio_path == "/path/to/audio.mp3"
        assert config.audio_prompt.start_time == 5.0
        assert config.audio_prompt.end_time == 20.0

    def test_parse_audio_prompt_dual(self, parser: ConfigParser) -> None:
        """Test parsing audio prompt in dual mode."""
        data = {
            "lyrics": {"sections": []},
            "audio_prompt": {
                "mode": "dual",
                "vocal_path": "/path/to/vocal.mp3",
                "instrumental_path": "/path/to/inst.mp3",
            },
        }

        config = parser.parse_dict(data)

        assert config.audio_prompt.mode == "dual"
        assert config.audio_prompt.vocal_path == "/path/to/vocal.mp3"
        assert config.audio_prompt.instrumental_path == "/path/to/inst.mp3"

    def test_default_values(self, parser: ConfigParser) -> None:
        """Test that default values are applied correctly."""
        data = {
            "lyrics": {"sections": []},
            "generation": {},  # Empty generation should use defaults
        }

        config = parser.parse_dict(data)

        assert config.generation.stage1_model == "m-a-p/YuE-s1-7B-anneal-en-cot"
        assert config.generation.stage2_model == "m-a-p/YuE-s2-1B-general"
        assert config.generation.max_new_tokens == 3000
        assert config.generation.repetition_penalty == 1.1
        assert config.generation.run_n_segments == 2
        assert config.generation.stage2_batch_size == 4
        assert config.generation.seed == 42


class TestParseFile:
    """Test parsing from file."""

    @pytest.fixture
    def parser(self) -> ConfigParser:
        """Create a parser instance."""
        return ConfigParser()

    def test_parse_basic_template(self, parser: ConfigParser) -> None:
        """Test parsing the basic template file."""
        template_path = Path(__file__).parent.parent / "yue_cli" / "templates" / "basic.yaml"
        if template_path.exists():
            config = parser.parse_file(template_path)
            assert config.metadata.name == "my-song"
            assert not config.is_variations_mode()

    def test_parse_variations_template(self, parser: ConfigParser) -> None:
        """Test parsing the variations template file."""
        template_path = Path(__file__).parent.parent / "yue_cli" / "templates" / "variations.yaml"
        if template_path.exists():
            config = parser.parse_file(template_path)
            assert config.metadata.name == "my-song"
            assert config.is_variations_mode()
            assert len(config.variations) == 2

    def test_file_not_found(self, parser: ConfigParser) -> None:
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/path/config.yaml")
