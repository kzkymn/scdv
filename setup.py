from setuptools import setup

setup(
    name="scdv",
    version="0.0.1",
    install_requires=[
        "gensim",
        "numpy",
        "sklearn"],
    extras_require={
        "dev": ["yellowbrick"]
    },
    entry_points={
    }
)
