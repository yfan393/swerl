"""
setup.py — SWE-RL Re-implementation
=====================================
Installs the package so all modules can be imported as:
    from data.fetch_gharchive import ...
    from agent.retriever import ...
    from reward.reward_fn import ...
    from training.grpo_train import ...
    from sft.sft_train import ...
    from evaluation.evaluate import ...
"""

from setuptools import find_packages, setup

setup(
    name="swe_rl_reimplement",
    version="0.1.0",
    description="SWE-RL at Small Scale: Partial re-implementation (CS 8803)",
    author="Yuanting Fan",
    author_email="yfan393@gatech.edu",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "trl>=0.11.0",
        "datasets>=2.21.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu>=1.8.0",
        "numpy>=1.26.0",
        "requests>=2.32.0",
        "tqdm>=4.66.0",
        "unidiff>=0.7.5",
        "tiktoken>=0.7.0",
        "tenacity>=8.5.0",
        "openai>=1.50.0",
        "pyyaml>=6.0.2",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "pre-commit"],
        "gpu": ["faiss-gpu>=1.8.0", "bitsandbytes>=0.43.0"],
        "wandb": ["wandb>=0.17.0"],
    },
)
