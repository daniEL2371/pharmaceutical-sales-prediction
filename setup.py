#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pandas>=1.1.0', 'numpy>=1.19.0', 'sklearn',
                'seaborn', 'matplotlib', 'plotly', 'scipy', 'dvc', 'mlflow']

test_requirements = ['pytest>=3', ]

setup(
    author="Daniel Zelalem",
    email="danielzelalemheru@gmail.com",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Machine learning approcah for rossman sales prediction",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='scripts, sales prediction, time series data, data exploration, random forest, linear regression, LSTM, RNN deep learning',
    name='pharmaceutical-sales-prediction',
    packages=find_packages(include=['scripts', 'scripts.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/daniEL2371/pharmaceutical-sales-prediction',
    version='0.1.0',
    zip_safe=False,
)
