import setuptools

setuptools.setup(
    name='PGrisk',
    version='0.1',
    description="risk analysis tools for load and production"
                "of wind and solar in power grid system",

    author='Xinshuo Yang',
    author_email='xy3134@princeton.edu',

    packages=['pgrisk'],

    python_requires='==3.8',
    install_requires=['numpy>1.21', 'pandas<2', 'scipy', 'dill', 'matplotlib',
                      'gurobipy'],
    )