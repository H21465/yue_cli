# yue_cli

YAML-based CLI interface for [YuE](https://github.com/multimodal-art-projection/YuE) music generation.

## Features

- YAML configuration instead of manually editing genre.txt and lyrics.txt
- Multiple variations from a single config (same lyrics, different genres/seeds)
- Dry-run mode for command verification
- Batch processing of multiple config files

## Installation

```bash
pip install -r yue_cli/requirements.txt
```

## Usage

### Run inference

```bash
python -m yue_cli.cli run config.yaml
```

Options:
- `-v, --variation NAME`: Run specific variation(s) only
- `-o, --output-dir PATH`: Override output directory
- `-n, --dry-run`: Show commands without executing

### List variations

```bash
# Single file
python -m yue_cli.cli list config.yaml

# Directory (all YAML files)
python -m yue_cli.cli list configs/

# Show infer.py commands
python -m yue_cli.cli list config.yaml --commands
```

### Create config template

```bash
python -m yue_cli.cli init --template variations -o my-song.yaml
```

### Validate config

```bash
python -m yue_cli.cli validate config.yaml
```

## Config Example

```yaml
version: "1.0"

metadata:
  name: "my-song"

lyrics:
  sections:
    - type: verse
      text: |
        First verse lyrics here
    - type: chorus
      text: |
        Chorus lyrics here

defaults:
  generation:
    stage1_model: "m-a-p/YuE-s1-7B-anneal-en-cot"
    stage2_model: "m-a-p/YuE-s2-1B-general"
    max_new_tokens: 3000
    run_n_segments: 2

variations:
  - name: "pop-v1"
    genre:
      tags: [pop, female, uplifting]
    generation:
      seed: 42

  - name: "pop-v2"
    genre:
      tags: [pop, female, uplifting]
    generation:
      seed: 123
```

## Docker

This CLI is designed to work inside the YuE Docker container. See the main project's Dockerfile for setup.

```bash
# Inside container
python -m yue_cli.cli run /workspace/configs/my-song.yaml
```

## License

MIT License - see [LICENSE](LICENSE) file.
