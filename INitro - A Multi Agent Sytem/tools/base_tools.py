"""Base Tools Implementation

Common tools and utilities that can be shared between agents.
"""


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool_func):
        """Register a new tool"""
        self.tools[name] = tool_func

    def get_tool(self, name):
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self):
        """List all available tools"""
        return list(self.tools.keys())


class BaseTool:
    """Base class for all tools"""

    def __init__(self, name):
        self.name = name

    def execute(self, *args, **kwargs):
        """Execute the tool"""
        raise NotImplementedError("Subclasses must implement execute method")
