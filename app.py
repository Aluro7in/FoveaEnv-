"""
app.py — Hugging Face Spaces Entry Point for FoveaEnv
This file is the main entry point for deploying to Hugging Face Spaces.
It imports the FastAPI app from server.py and runs it on port 7860.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import the FastAPI app from server.py
from server.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info"
    )
