from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rewardguard",
    version="1.0.3",
    author="RewardGuard Team",
    author_email="giovan@rewardguard.dev",
    description="AI alignment and reward balance analysis for reinforcement learning systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rewardguard.dev",
    project_urls={
        "Documentation": "https://docs.rewardguard.dev",
        "Website": "https://rewardguard.dev",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No external dependencies - uses only Python standard library
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    keywords=[
        "reinforcement-learning",
        "AI",
        "machine-learning",
        "reward-hacking",
        "AI-alignment",
        "AI-safety",
        "RL",
        "training-analysis",
        "reward-balance",
    ],
    entry_points={
        "console_scripts": [
            "rewardguard=rewardguard.cli:main",
        ],
    },
)
