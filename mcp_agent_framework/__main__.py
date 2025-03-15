"""
Main entry point for MCP Agent Framework when run as a package.

This module allows the framework to be run directly as a package using
`python -m mcp_agent_framework`.
"""

import sys
import os
import argparse
from .framework import AgentFramework, run_cli

def main():
    """
    Process command line arguments and run the appropriate command.
    """
    parser = argparse.ArgumentParser(description="MCP Agent Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Start the interactive CLI")
    cli_parser.add_argument("--config", help="Path to configuration file")
    cli_parser.add_argument("--api-key", help="API key for language models")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a task and exit")
    submit_parser.add_argument("task", help="Task description")
    submit_parser.add_argument("--config", help="Path to configuration file")
    submit_parser.add_argument("--api-key", help="API key for language models")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for task completion")
    submit_parser.add_argument("--timeout", type=float, help="Timeout for waiting (in seconds)")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "cli":
        # Start interactive CLI
        framework = AgentFramework(
            api_key=args.api_key,
            config_path=args.config
        )
        framework.start_cli()
    
    elif args.command == "submit":
        # Submit task and exit
        import asyncio
        
        async def submit_task():
            framework = AgentFramework(
                api_key=args.api_key,
                config_path=args.config
            )
            
            # Initialize framework
            await framework.initialize()
            
            # Submit task
            task_id = await framework.submit_task(args.task)
            print(f"Task submitted with ID: {task_id}")
            
            # Wait for completion if requested
            if args.wait:
                print("Waiting for task completion...")
                task_info = await framework.wait_for_task(task_id, args.timeout)
                
                if task_info and task_info["status"] == "COMPLETED":
                    print("\nTask completed successfully!")
                    print(f"\nResult:\n{task_info['result']}")
                    return 0
                elif task_info and task_info["status"] == "FAILED":
                    print(f"\nTask failed: {task_info['error']}")
                    return 1
                else:
                    print("\nTask did not complete within the timeout period.")
                    return 1
            
            return 0
        
        sys.exit(asyncio.run(submit_task()))
    
    elif args.command == "version":
        # Show version information
        from . import __version__
        print(f"MCP Agent Framework v{__version__}")
    
    else:
        # No command specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
