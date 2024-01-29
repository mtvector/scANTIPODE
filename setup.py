from setuptools import setup, find_packages

setup(
    name="scantipode",
    version="0.1",
    packages=find_packages(),
    #First install cuda-specific pytorch using conda
    install_requires=[
        'scanpy',
        'scvi-tools',
        'pyro-ppl'
]
    ],
    # Additional metadata about your package.
)

