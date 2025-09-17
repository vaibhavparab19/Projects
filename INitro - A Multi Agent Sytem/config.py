import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the multi-agent system"""

    # System settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Agent settings
    MAX_AGENTS = int(os.getenv("MAX_AGENTS", "5"))
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "30"))

    # Communication settings
    MESSAGE_QUEUE_SIZE = int(os.getenv("MESSAGE_QUEUE_SIZE", "100"))

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    # LangChain Configuration
    LANGCHAIN_TRACING_V2 = os.getenv(
        "LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-system")

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if cls.MAX_AGENTS <= 0:
            raise ValueError("MAX_AGENTS must be positive")
        if cls.AGENT_TIMEOUT <= 0:
            raise ValueError("AGENT_TIMEOUT must be positive")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True

    @classmethod
    def get_openai_config(cls):
        """Get OpenAI configuration dictionary"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
        }


# System Configuration
SYSTEM_CONFIG = {
    "name": "Multi-Agent System",
    "version": "1.0.0",
    "description": "Multi-agent system with thinking node coordination",
}

# Agent Configuration
AGENT_CONFIG = {
    "agent1": {"name": "Agent1", "enabled": True, "max_iterations": 100},
    "agent2": {"name": "Agent2", "enabled": True, "max_iterations": 100},
}

# Thinking Node Configuration
THINKING_NODE_CONFIG = {
    "coordination_mode": "sequential",
    "timeout": 30,
    "retry_attempts": 3,
}

# Tools Configuration
TOOLS_CONFIG = {
    "communication": {"enabled": True, "message_queue_size": 1000},
    "shared_utilities": {"enabled": True},
}

# Flow Configuration (based on the flowchart)
FLOW_CONFIG = {
    "start_node": "thinking_node",
    "exit_conditions": [
        "task_completed",
        "max_iterations_reached",
        "error_threshold_exceeded",
    ],
}
