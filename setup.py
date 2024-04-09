from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="antipode",
    author='Matthew Schmitz',
    description="Simultaneous integration, differential expression and clustering of single cell RNAseq data using pyro-powered variational inference.",
    long_description=long_description,
    long_description_content_type='text/markdown',  # Ensure your README is displayed correctly on PyPI
    version="0.1",
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'scanpy',
        'scvi-tools',
        'pymde',
        'pyro-ppl',
        'seaborn',
        'scikit-learn',
        'PyComplexHeatmap'
        ]
)
