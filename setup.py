try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'T1 fitting package',
    'author': 'Andrew Curtis',
    'url': 'tbd. camhpet.ca',
    'author_email': 'andrew.curtis@gmail.com',
    'version': '0.1',
    'install_requires': ['numpy','scipy','nose','nibabel'],
    'name': 't1fitter'
}

setup(**config)

