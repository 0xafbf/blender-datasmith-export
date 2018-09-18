
from distutils.core import setup, Extension

module_datasmith = Extension('datasmith', sources=['datasmith.c'])

setup(
    name='bdatasmith',
    version='4.20',
    description='demo datasmith package',
    ext_modules=[module_datasmith]
)