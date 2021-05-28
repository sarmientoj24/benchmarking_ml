from setuptools import find_namespace_packages, setup

setup(
    name="benchmarking_ml",
    version="0.1.0",
    description="CS 250: Advanced Operating Systems",
    author="James-Andrew Sarmiento",
    packages=find_namespace_packages(include=["src*"]),
    python_requires="~=3.6.9",
    install_requires=[
        "numpy~=1.18.5",
        "torch~=1.8.1",
        "scikit-learn~=0.24.2",
        "wandb~=0.10.30",
        "scipy==1.5.4",
        "tqdm~=4.61.0",
        "torchvision~=0.9.1",
        "nltk~=3.6.2",
        "pandas~=1.1.5"
    ],
)