from setuptools import setup, find_packages

setup(
    name="mcp_agent_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mcp>=1.2.0",  # Model Context Protocol package
        "httpx>=0.24.0",  # Async HTTP client
        "rich>=13.3.5",  # Rich terminal output
        "pydantic>=2.0.0",  # Data validation
        "asyncio>=3.4.3",  # Async I/O
    ],
    author="Zhang Jianfeng",
    author_email="jianfeng.zhang@example.com",
    description="A framework for creating, orchestrating, and managing hierarchical networks of AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maxoreric/mcp-agent-framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
