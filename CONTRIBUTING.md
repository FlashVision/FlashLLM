# Contributing to FlashLLM

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashLLM.git
cd FlashLLM
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashllm/`
4. Run tests: `flashllm check`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Solution

1. Create `flashllm/solutions/your_solution.py`
2. Follow the existing pattern: accept `model_id` or `predictor`
3. Implement the main method (e.g., `chat()`, `summarize()`, `generate()`)
4. Implement `reset()` for stateful solutions
5. Add to `flashllm/solutions/__init__.py`

## Adding a New Training Method

1. Create `flashllm/training/your_method.py`
2. Inherit from the base trainer pattern
3. Implement `train_step()` and `compute_loss()`
4. Register in `flashllm/registry.py`
5. Add to `flashllm/training/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add DPO training implementation`
- `Fix KV cache memory leak in streaming`
- `Update README with quantization examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashllm settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
