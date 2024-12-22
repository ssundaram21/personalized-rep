import setuptools

setuptools.setup(
    name="personalized_synthetic",
    version="0.1.0",
    description="Utilities for launching jobs on the vision cluster",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# "scipy==1.9.2",
# "timm==0.6.12",
