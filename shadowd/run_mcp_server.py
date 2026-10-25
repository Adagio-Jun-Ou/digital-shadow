#!/usr/bin/env python3
"""
Shadow AI MCP Server Standalone Runner

Run the MCP server independently or as part of the daemon.

Usage:
    python -m shadowd.run_mcp_server --transport stdio
    python -m shadowd.run_mcp_server --transport streamable-http --port 8000
    python -m shadowd.run_mcp_server --transport sse --port 8080
"""

import sys
import os
import logging
from pathlib import Path
import argparse
import asyncio

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.mcp_server_enhanced import EnhancedMCPServer, HAS_MCP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    if not HAS_MCP:
        logger.error(
            "MCP package not installed. Install with:\n"
            "  pip install mcp"
        )
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Shadow AI MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with stdio (for direct process communication)
  python -m shadowd.run_mcp_server --transport stdio
  
  # Run with streamable HTTP (for web access)
  python -m shadowd.run_mcp_server --transport streamable-http --port 8000
  
  # Run with SSE (for browser clients)
  python -m shadowd.run_mcp_server --transport sse --port 8080

For more information about MCP:
  https://modelcontextprotocol.io
        """
    )
    
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="Transport protocol (default: streamable-http)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transports (default: 8000)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transports (default: localhost)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory for Shadow AI storage"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Shadow AI MCP Server 2.0.0"
    )
    
    args = parser.parse_args()
    
    # Create MCP server
    logger.info("Starting Shadow AI MCP Server...")
    logger.info(f"Transport: {args.transport}")
    
    if args.transport == "stdio":
        logger.info("Running in stdio mode (for direct process communication)")
    else:
        logger.info(f"Listening on {args.host}:{args.port}")
    
    server = EnhancedMCPServer(
        name="Shadow AI",
        version="2.0.0",
        data_dir=args.data_dir
    )
    
    try:
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down MCP server...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()