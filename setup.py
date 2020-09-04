from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tysserand',
    version='0.0.1',
    description='A Python package for spatial network construction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/AlexCoul/tysserand",
    author='Alexis Coullomb',
    author_email='alexis.coullomb.pro@gmail.com',
    license='BSD 3',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Operating System :: OS Independent'],
    packages=find_packages(exclude=['build', 'docs', 'templates', 'data']),
    include_package_data=True,
    install_requires=['matplotlib>=2.2.3', 'numpy', 'seaborn',
                      'PySAL>=2.0,<2.1', 'pandas>=0.23,<0.24','scipy',
                      'Pillow', 'opencv-python', 'scikit-image',
                      'scikit-learn',
                     ],
    keywords = 'spatial networks cells sociology econometrics'
)
