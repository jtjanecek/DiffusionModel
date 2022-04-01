import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffusion_model",
    version="0.0.1",
    author="John Janecek",
    author_email="janecektyler@gmail.com",
    description="Python wrapper and models for Diffusion modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtjanecek/DiffusionModel",
    project_urls={
        "Bug Tracker": "https://github.com/jtjanecek/DiffusionModel/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
	install_requires=['pandas', 'numpy', 'h5py', 'arviz'],
	package_data = {'': ['*.jags', '*.params']},
	package_dir={"": "src"},
	packages=setuptools.find_packages(where='src'),
    python_requires=">=3.7",
)
