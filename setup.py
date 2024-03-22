from distutils.core import setup
from setuptools import find_packages

NAME = 'rag'
VERSION = '0.1.1'
LICENSE = 'MIT'
DESCRIPTION = 'Virtual Sale Assistant'

with open('README.md', encoding="utf-8") as file: 
    description = file.read()
setup(
    name=NAME, 
    version=VERSION, 
    packages=find_packages(where="./rag"), 
    package_dir={"": "rag"},
    license=LICENSE, 
    zip_safe=True, 
    description=DESCRIPTION, 
    author='Hung Ng. Quoc', 
    author_email='hungnq131002@gmail.com', 
    url='https://github.com/hungnq1310/rag_pdf_services', 
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 1 - Planning', 
        'Intended Audience :: Developers', 
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9'
    ],
)