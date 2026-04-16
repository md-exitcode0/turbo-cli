from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

setup(
    name="turbo-cli",
    version="1.1.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"turbo": ["data/*", "bin/*.exe"]},
    include_package_data=True,
    install_requires=["rich>=13.0", "questionary>=2.0"],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["turbo=turbo.cli:main"]},
    license="Apache-2.0",
    zip_safe=False,
)
