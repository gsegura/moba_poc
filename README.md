# MoBA Deep Research POC

Implementation of a Deep Research agent using Mixture of Block Attention (MoBA) and SSD-based KV Cache.

## Installation

This project uses UV as the package manager. To install:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Project Structure

- `moba_research/`: Main package directory
  - `attention.py`: MoBA attention implementation
  - `cache.py`: SSD-based KV cache
  - `preprocessing.py`: Text processing and embedding
  - `search.py`: Search API integration

## Usage

See ROADMAP.md for implementation details and project phases.

## Development

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Format code:
   ```bash
   black moba_research tests
   isort moba_research tests
   ```
