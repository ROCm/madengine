# MADEngine Developer Guide

This guide covers development setup, coding standards, and contribution guidelines for MADEngine.

## Quick Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd madengine

# Development setup
pip install -e ".[dev]"
pre-commit install
```

## Modern Python Packaging

This project follows modern Python packaging standards:

- **`pyproject.toml`** - Single configuration file for everything
- **No requirements.txt** - Dependencies defined in pyproject.toml
- **Hatchling** - Modern build backend
- **Built-in tool configuration** - Black, pytest, mypy, etc. all configured in pyproject.toml

### Installation Commands

```bash
# Production install
pip install .

# Development install (includes dev tools)
pip install -e ".[dev]"

# Build package
python -m build  # requires: pip install build
```

## Development Workflow

### 1. Code Formatting and Linting

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/madengine

# Run all tools at once
pre-commit run --all-files
```

### 2. Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=madengine --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run tests with specific marker
pytest -m "not slow"
```

### 3. Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Install hooks (already done in setup)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Coding Standards

### Python Code Style

- Follow PEP 8 style guide
- Use Black for automatic formatting (line length: 88)
- Sort imports with isort
- Maximum cyclomatic complexity: 10
- Use type hints where possible

### Documentation

- All public functions and classes must have docstrings
- Follow Google-style docstrings
- **Primary documentation is in README.md** - Keep it comprehensive and up-to-date
- Document any new configuration options in the README
- For major features, include examples in the appropriate README sections
- Update CLI documentation when adding new commands
- Include deployment scenarios for distributed features

### Error Handling

- Use proper logging instead of print statements
- Handle exceptions gracefully
- Provide meaningful error messages
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

### Testing

- Write tests for new functionality
- Maintain test coverage above 80%
- Use meaningful test names
- Follow AAA pattern (Arrange, Act, Assert)

## Code Organization

```
src/madengine/
├── __init__.py          # Package initialization
├── mad.py              # Main CLI entry point
├── core/               # Core functionality
├── db/                 # Database operations
├── tools/              # CLI tools
├── utils/              # Utility functions
└── scripts/            # Shell scripts and tools
```

## Adding New Features

### Documentation Guidelines

MADEngine uses a centralized documentation approach:

- **README.md** is the primary documentation source containing:
  - Installation and quick start guides
  - Complete CLI reference
  - Distributed execution workflows
  - Configuration options and examples
  - Deployment scenarios
  - Contributing guidelines

- **Additional documentation** should be minimal and specific:
  - `DEVELOPER_GUIDE.md` - Development setup and coding standards
  - `docs/how-to-*.md` - Specific technical guides
  - `CHANGELOG.md` - Release notes and changes

When adding features:
1. Update the relevant README.md sections
2. Add CLI examples if applicable
3. Include configuration options
4. Document any new MAD package integration patterns
5. Add deployment scenarios for distributed features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature**
   - Write the code following our standards
   - Add comprehensive tests
   - Update documentation

3. **Test your changes**
   ```bash
   pytest --cov=madengine
   pre-commit run --all-files
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Submit a pull request**
   - Ensure all CI checks pass
   - Write a clear description
   - Request appropriate reviewers

## Environment Variables

MADEngine uses several environment variables for configuration:

- `MODEL_DIR`: Location of models directory
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAD_VERBOSE_CONFIG`: Enable verbose configuration logging
- `MAD_AWS_S3`: AWS S3 credentials (JSON)
- `NAS_NODES`: NAS configuration (JSON)
- `PUBLIC_GITHUB_ROCM_KEY`: GitHub token (JSON)

## Common Tasks

### Adding a New CLI Command

1. Create a new module in `src/madengine/tools/`
2. Add the command handler in `mad.py`
3. Update the argument parser
4. Add tests in `tests/`
5. Update documentation

### Adding Dependencies

1. Add to `pyproject.toml` under `dependencies` or `optional-dependencies`
2. Update setup.py if needed for legacy compatibility
3. Run `pip install -e ".[dev]"` to install
4. Update documentation if the dependency affects usage

### Debugging

- Use the logging module instead of print statements
- Set `LOG_LEVEL=DEBUG` for verbose output
- Use `MAD_VERBOSE_CONFIG=true` for configuration debugging

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with new features, changes, and fixes
3. Ensure README.md reflects all current functionality
4. Create a release tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
5. Push tag: `git push origin v1.0.0`
6. Build and publish: `python -m build`

### Documentation Updates for Releases

- Verify README.md covers all new features
- Update CLI examples if commands have changed
- Ensure configuration examples are current
- Add any new deployment scenarios
- Update MAD package integration examples if applicable

## Troubleshooting

### Common Issues

1. **Import errors**: Check if package is installed in development mode
2. **Test failures**: Ensure all dependencies are installed
3. **Pre-commit failures**: Run `black src/ tests/` and `isort src/ tests/` to fix formatting issues
4. **Type checking errors**: Add type hints or use `# type: ignore` comments

### Getting Help

- **Start with README.md** - Comprehensive documentation covering most use cases
- Check existing issues in the repository
- Review specific guides in `docs/` directory for advanced topics
- Contact the development team
- For CLI questions, refer to the CLI reference section in README.md
- For distributed execution, see the distributed workflows section in README.md

## Performance Considerations

- Profile code for performance bottlenecks
- Use appropriate data structures
- Minimize I/O operations
- Cache expensive computations when possible
- Consider memory usage for large datasets

## Security Guidelines

- Never commit credentials or secrets
- Use environment variables for sensitive configuration
- Validate all user inputs
- Follow secure coding practices
- Keep dependencies updated
