"""
Setup configuration for Ontario Power Plant Optimization project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ontario-power-optimization",
    version="0.1.0",
    author="Ontario Power Planning Team",
    description="Multi-objective optimization for Ontario power plant planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ontario-power-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "jupyter>=1.0.0",
        ],
    },
)
