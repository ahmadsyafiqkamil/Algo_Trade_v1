from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algo-trade-v1",
    version="0.1.0",
    author="Ahmad Syafiq Kamil",
    author_email="your.email@example.com",  # Ganti dengan email Anda
    description="A Python library for algorithmic trading analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Algo_Trade_v1",  # Ganti dengan URL GitHub Anda
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "yfinance",
        "ta",
        "requests",
        "beautifulsoup4",
    ],
) 