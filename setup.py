"""
Setup script for Emotion-Weighted Memory LLM package
"""

from setuptools import setup, find_packages

# READ README FOR LONG DESCRIPTION
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="emotion-weighted-memory-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Plug-and-Play Cognitive Overlay for Pretrained Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emotion-weighted-memory-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "qdrant-client>=1.6.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "dataclasses-json>=0.6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch[cu118]>=2.0.0",  # CUDA 11.8 SUPPORT
        ],
        "full": [
            "clip",  # FOR VISION ENCODING
            "whisper",  # FOR AUDIO ENCODING
            "openai",  # FOR API MODELS
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-llm=overlay.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "artificial-intelligence",
        "machine-learning",
        "deep-learning",
        "language-models",
        "emotions",
        "memory",
        "cognitive-architecture",
        "neuroscience",
        "psychology",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/emotion-weighted-memory-llm/issues",
        "Source": "https://github.com/yourusername/emotion-weighted-memory-llm",
        "Documentation": "https://github.com/yourusername/emotion-weighted-memory-llm#readme",
    },
)
