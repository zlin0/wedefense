from setuptools import setup, find_packages

requirements = [
    "torch>=1.12.0",
    "torchaudio>=0.12.0",
]

setup(
    name="wedefense",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wedefense = wedefense.cli.main:main",
        ]
    },
)
