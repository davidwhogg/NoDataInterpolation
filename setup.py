from setuptools import setup, find_packages

setup(
    name="ndi",
    version="0.0.1",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    install_requires=["numpy"]
)
