"""Claude Code Proxy

A proxy server that enables Claude Code to work with OpenAI-compatible API providers.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv


def _parse_env_file_from_args(args) -> Optional[str]:
    """Parse command-line args for env file selection.

    Supported flags:
    - --env-file <path> or --env-file=path
    - --env <name> or --env=name -> loads .env.<name>
    """
    env_path: Optional[str] = None
    for i, arg in enumerate(args):
        if arg == "--env-file" and i + 1 < len(args):
            env_path = args[i + 1]
            break
        elif arg.startswith("--env-file="):
            env_path = arg.split("=", 1)[1]
            break
        elif arg == "--env" and i + 1 < len(args):
            env_name = args[i + 1]
            env_path = f".env.{env_name}"
            break
        elif arg.startswith("--env="):
            env_name = arg.split("=", 1)[1]
            env_path = f".env.{env_name}"
            break
    return env_path


def _load_selected_env_file():
    # Priority: explicit env var -> CLI flags -> default .env
    explicit = os.environ.get("CLAUDE_PROXY_ENV_FILE")
    chosen = explicit or _parse_env_file_from_args(sys.argv)

    if chosen:
        # When an explicit env file is chosen, allow it to override existing envs.
        load_dotenv(dotenv_path=chosen, override=True)
        # Minimal notice for visibility; avoid printing secrets.
        print(f"Using env file: {chosen}")
    else:
        # Default behavior: load .env if present, without overriding existing shell envs.
        load_dotenv()


# Load environment variables early when the package is imported
_load_selected_env_file()

__version__ = "1.0.0"
__author__ = "Claude Code Proxy"
