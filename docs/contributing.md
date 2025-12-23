# Contributing to madengine

Thank you for your interest in contributing! We welcome all contributions, whether they are bug fixes, new features, or improvements to documentation.

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/madengine.git
cd madengine
```

### 2. Setup Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

## Development Workflow

### Making Changes

1. **Implement your changes** in the appropriate files
2. **Write tests** for new functionality (place in `tests/` directory)
3. **Update documentation** if needed
4. **Follow code standards** (see below)

### Code Standards

- **Style**: Black formatting (88 character line length)
- **Imports**: Organized with isort
- **Type Hints**: Add type hints for all public functions
- **Docstrings**: Use Google-style docstrings
- **Testing**: Maintain 95%+ test coverage for new code

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/madengine --cov-report=html

# Run specific test file
pytest tests/test_cli.py

# Run tests matching pattern
pytest -k "test_build"
```

### Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/madengine

# Run all quality checks (if pre-commit installed)
pre-commit run --all-files
```

## Commit Guidelines

Use conventional commit format:

```bash
# Good commit messages
git commit -m "feat(cli): add SLURM runner support"
git commit -m "fix(k8s): handle connection timeouts gracefully"
git commit -m "docs: update deployment examples"
git commit -m "test: add integration tests for build command"

# Commit types
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# test: Test additions/changes
# refactor: Code refactoring
# style: Code style changes (formatting, etc.)
# perf: Performance improvements
# chore: Build process or auxiliary tool changes
```

## Submitting Changes

### 1. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to the [madengine repository](https://github.com/ROCm/madengine)
2. Click "New Pull Request"
3. Select your fork and branch
4. Provide a clear description:
   - What changes were made
   - Why the changes were needed
   - Any related issues (use `Fixes #123` to auto-close issues)

### 3. Pull Request Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code follows style guidelines (`black`, `isort`, `flake8`)
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch

## Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainers review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

## Areas for Contribution

### High Priority

- Additional deployment backends
- Performance optimizations
- Enhanced error messages
- Test coverage improvements

### Medium Priority

- CLI enhancements
- Documentation improvements
- Monitoring and observability
- Configuration simplification

### Good First Issues

Look for issues labeled `good-first-issue` on GitHub.

## Development Tips

### Project Structure

```
madengine/
├── src/madengine/
│   ├── cli/              # CLI commands
│   ├── orchestration/    # Build and run orchestrators
│   ├── deployment/       # K8s and SLURM deployment
│   ├── execution/        # Container execution
│   ├── core/            # Core utilities
│   └── utils/           # Helper functions
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Example configurations
```

### Testing Philosophy

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: End-to-end workflow testing
- **Fixtures**: Use pytest fixtures for common test data
- **Mocking**: Mock external dependencies (Docker, K8s API, etc.)

### Debugging

```bash
# Run with verbose logging
madengine run --tags model --verbose

# Keep containers alive for debugging
madengine run --tags model --keep-alive

# Use Python debugger
python -m pdb -m madengine.cli.app run --tags model
```

## Getting Help

- **GitHub Issues**: https://github.com/ROCm/madengine/issues
- **Discussions**: https://github.com/ROCm/madengine/discussions
- **Documentation**: [docs/](.)

## Code of Conduct

Be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## Recognition

Contributors are recognized in:
- **CHANGELOG.md**: All contributions documented
- **GitHub Contributors**: Automatic recognition
- **Release Notes**: Major contributions highlighted

Thank you for contributing to madengine!

