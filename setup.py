from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pulse_ops",
    version="1.0.0",
    author="PULSE-OPS Team",
    author_email="pulse-ops@example.com",
    description="Autonomous MLOps Platform — end-to-end ML lifecycle management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PULSE-OPS/PULSE-OPS",
    project_urls={
        "Bug Tracker": "https://github.com/PULSE-OPS/PULSE-OPS/issues",
        "Documentation": "https://github.com/PULSE-OPS/PULSE-OPS/wiki",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "black>=24.4.2",
            "flake8>=7.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pulse-train=orchestration.training_flow:main",
            "pulse-drift=orchestration.drift_flow:main",
            "pulse-retrain=orchestration.retraining_flow:main",
            "pulse-serve=serving.fastapi_server:main",
            "pulse-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "prometheus/*.yml", "grafana/**/*.json"],
    },
)
