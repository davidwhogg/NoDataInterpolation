from setuptools import setup, find_packages

setup(
    name="NoDataInterpolation",
    version="0.0.1",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    install_requires=["numpy"]
)
