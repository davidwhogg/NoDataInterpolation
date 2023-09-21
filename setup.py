from setuptools import setup, find_packages

setup(
    name="ndi",
    version="0.0.1",
    packages=find_packages(where="code"),
    install_requires=["numpy"]
)
