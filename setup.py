from setuptools import setup, find_packages

setup(
    name = "STRIKE_YOLO",
    version = "0.1",
    packages = find_packages(),
    install_requires= [
        'ultralytics==8.0.154',
        'adjustText==0.8',
        'numpy',
        'pandas',
        'scipy'
    ],
)