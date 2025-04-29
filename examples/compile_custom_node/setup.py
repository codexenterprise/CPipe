# python setup.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "./custom_node.py",
    ]
    ),
)



