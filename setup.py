#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="digital-shadow",
    version="0.1.0",
    description="Your personal AI digital twin",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "httpx>=0.25.0",
        "pydantic>=2.5.0",
        "watchfiles>=0.21.0",
        "python-multipart>=0.0.6",
    ],
    entry_points={
        "console_scripts": [
            "shadowd=shadowd.shadowd:run",
            "summon=summon.summon_cli:main",
        ],
    },
    python_requires=">=3.10",
)