from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("c:/Users/Julian/Documents/AP&AM22-23/Q3-Q4/Advanced Modeling/HarmonicsApproach/Attempt/Example.pyx"))