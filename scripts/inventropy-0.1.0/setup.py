"""
Setup script for inventropy package
For backward compatibility with older pip versions
"""

from setuptools import setup

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inventropy",
    version="0.1.0",
    author="Haoyi Song",
    author_email="haoyiso@umich.edu",
    description="A Python package for computing inverse entropy in language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/inventropy",
    packages=["inventropy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai==0.28.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "sentence-transformers>=2.0.0",
        "transformers>=4.20.0",
        "scipy>=1.7.0",
        "sentencepiece>=0.1.96",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="inverse entropy language models NLP machine learning",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/inventropy/issues",
        "Documentation": "https://github.com/yourusername/inventropy#readme",
        "Source Code": "https://github.com/yourusername/inventropy",
    },
)
