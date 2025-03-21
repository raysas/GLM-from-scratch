from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='GLM-from-scratch',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Rayane Adam',
    author_email='rayane.s.adam@gmail.com',
    description='A python implementation of Generalized Linear Models from scratch',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/raysas/GLM-from-scratch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
