import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timesweeper",
    version="0.9",
    author="Logan Whitehouse",
    author_email="lswhiteh@unc.edu",
    description="A tool for detecting positive selective sweeps using time-series genomcis data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SchriderLab/timeSeriesSweeps",
    project_urls={"Issues": "https://github.com/SchriderLab/timeSeriesSweeps/issues",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "timesweeper"},
    packages=setuptools.find_packages(where="timesweeper"),
    python_requires=">=3.6",
)
