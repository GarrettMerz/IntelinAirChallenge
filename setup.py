"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    name='IntelinAirChallenge',  # Required
    version='1.0.0',  # Required
    description='Semantic segmentation of fields across time',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/GarrettMerz/IntelinAirChallenge',  # Optional
    author='Garrett Merz',  # Optional
    author_email='garrettwmerz@gmail.com',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=['numpy','pandas', 'matplotlib', 'opencv-python', 'tensorflow >= 2.2.0', 'keras >= 2.4.0'],
    project_urls={ 
        'AWS repo of data': 'https://registry.opendata.aws/intelinair_longitudinal_nutrient_deficiency/'
    },
    entry_points={
        'console_scripts': [
        'train_model=src.train_model:train_model',
        'run_model_on_test_set=src.run_model_on_test_set:run_model_on_test_set',
        'inferencing=src.inferencing:inferencing'
        ]
    }
)
