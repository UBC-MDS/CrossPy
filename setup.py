from distutils.core import setup

setup(
    name='CrossPy',
    version='0.1dev',
    packages=['CrossPy',],
    license='LICENSE.txt',
    description='Cross-validation',
    long_description=open('README.md').read(),
    author = ['Nazli Ozum Kafaee', 'Daniel Raff', 'Shun Chi'],
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "pytest",
    ]
)
