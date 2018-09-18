
from distutils.core import setup, Extension

module_datasmith = Extension('datasmith', sources=['datasmith.cpp'])

setup(
    name='datasmith',
    version='4.20',
    description='demo datasmith package',
    ext_modules=[module_datasmith],
    url='https://gitlab.com/boterock/blender-datasmith',
    author='Andres Botero',
    author_email='boterock@gmail.com',
)