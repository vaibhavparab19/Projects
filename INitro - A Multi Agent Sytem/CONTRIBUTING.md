# Contributing to INitro Multi-Agent System

We welcome contributions to the INitro Multi-Agent System! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key
- Basic understanding of LangChain and multi-agent systems

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/RPI-GROUP-AI-Lab/ai-poc-iNitro-iNitroAgenticDecisioning.git
   cd ai-poc-iNitro-iNitroAgenticDecisioning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests**
   ```bash
   python -m pytest
   python test_agent_workflow.py
   ```

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest
   python test_agent_workflow.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** where appropriate
- Write **docstrings** for all classes and functions
- Keep line length under **88 characters** (Black formatter)

### Code Structure

```python
"""Module docstring describing the purpose."""

from typing import Dict, List, Optional
import logging

from agents.base_agent import BaseAgent


class ExampleAgent(BaseAgent):
    """Example agent class with proper documentation.
    
    Args:
        config: Configuration dictionary
        llm_service: LLM service instance
    """
    
    def __init__(self, config: Dict, llm_service: Optional[object] = None):
        super().__init__()
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
    
    def process(self, data: List[Dict]) -> Dict:
        """Process input data and return results.
        
        Args:
            data: List of data dictionaries to process
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ValueError: If data format is invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
            
        # Implementation here
        return {"status": "success", "results": []}
```

### Testing Guidelines

- Write **unit tests** for all new functions
- Use **pytest** for testing framework
- Aim for **80%+ code coverage**
- Include **integration tests** for agent workflows

```python
import pytest
from agents.example_agent import ExampleAgent


class TestExampleAgent:
    """Test suite for ExampleAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"model": "gpt-3.5-turbo"}
        self.agent = ExampleAgent(self.config)
    
    def test_process_valid_data(self):
        """Test processing with valid data."""
        data = [{"id": 1, "name": "test"}]
        result = self.agent.process(data)
        assert result["status"] == "success"
    
    def test_process_empty_data(self):
        """Test processing with empty data raises ValueError."""
        with pytest.raises(ValueError):
            self.agent.process([])
```

## 🏗️ Architecture Guidelines

### Adding New Agents

1. **Inherit from BaseAgent**
2. **Implement required methods**
3. **Add to workflow configuration**
4. **Include comprehensive tests**
5. **Update documentation**

### Adding New Tools

1. **Inherit from BaseTool**
2. **Implement execute method**
3. **Add error handling**
4. **Register with agent system**
5. **Add usage examples**

### Memory System Integration

- Use **MemorySystem** for persistent storage
- Implement **proper cleanup** methods
- Consider **memory efficiency**
- Add **retrieval capabilities**

## 📚 Documentation

### Code Documentation

- **Docstrings**: All public methods and classes
- **Type hints**: Function parameters and return types
- **Inline comments**: Complex logic explanation
- **README updates**: New features and changes

### API Documentation

- **Endpoint descriptions**: Purpose and usage
- **Request/response examples**: JSON schemas
- **Error codes**: Possible error responses
- **Authentication**: Required credentials

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=agents --cov=tools

# Run specific test file
python -m pytest tests/test_agents.py

# Run integration tests
python test_agent_workflow.py
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **End-to-End Tests**: Complete system testing
- **Performance Tests**: Load and stress testing

## 🚀 Deployment

### Development Deployment

```bash
# Local development server
python app.py

# With debug mode
FLASK_ENV=development python app.py
```

### Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t initro-system .
docker run -p 5000:5000 initro-system
```

## 🔍 Code Review Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No sensitive data in commits
- [ ] Commit messages are descriptive

### Review Criteria

- **Functionality**: Does it work as intended?
- **Performance**: Is it efficient?
- **Security**: Are there any vulnerabilities?
- **Maintainability**: Is the code readable and well-structured?
- **Testing**: Are there adequate tests?

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(agents): add customer segmentation agent
fix(memory): resolve memory leak in conversation storage
docs(readme): update installation instructions
test(agents): add unit tests for reflect agent
```

## 🆘 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check README and inline docs
- **Code Examples**: Review existing implementations

## 📋 Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimization**
- **Error handling improvements**
- **Test coverage expansion**
- **Documentation enhancements**

### Medium Priority
- **New agent implementations**
- **Additional export formats**
- **UI/UX improvements**
- **Integration with other AI services**

### Low Priority
- **Code refactoring**
- **Additional utility functions**
- **Example applications**
- **Deployment automation**

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to INitro Multi-Agent System! 🚀**