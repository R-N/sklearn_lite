import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sklearn',  
    version='1.7dev0',
    author="Scikit-Learn",
    description="Scikit-Learn lite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R-N/sklearn_lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.19.5",
        "joblib>=1.2.0",
        "threadpoolctl>=3.1.0",
    ],
)