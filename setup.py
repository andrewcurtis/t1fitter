try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'T1 fitting package',
    'author': 'Andrew Curtis',
    'url': '',
    'author_email': 'andrew.curtis@gmail.com',
    'version': '0.1',
    'install_requires': ['numpy','scipy','nose','nibabel','sh','traits'],
    'name': 't1fitter',
    'entry_points': {
            'console_scripts': [
                't1fit = t1fitter.run_t1fit:main'
            ]},
}

setup(**config)
