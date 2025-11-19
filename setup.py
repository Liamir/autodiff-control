from setuptools import setup, find_packages

setup(
    name="rpa_control",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "notebook",
        "ipykernel",
        "statsmodels",
        "fire",
        "torch",
        "torchdiffeq",
        "gymnasium",
        "stable-baselines3",
        "pytest",
    ],
)
