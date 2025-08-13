from setuptools import setup, find_packages

setup(
    name="panorama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.12.0",
        "numpy>=1.20.0",
        "cffi>=1.15.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "panorama=panorama.main:main",
        ],
    },
)