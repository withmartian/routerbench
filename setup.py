from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="routerbench",
    version="0.1",
    description="Code for the RouterBench paper",
    author="Jason Hu, Jacob Bieker",
    author_email="jason@withmartian.com",
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
)
