from setuptools import setup

setup(name='DeepAutoCov',
    version='0.1.0',
    #description='''''',
    #url='',
    #author='',
    #author_email='',
    #packages=[''],
    py_modules = [],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=[
        'pandas==2.1.2', 'numpy==1.26.1', 'matplotlib==3.8.1',
        'scipy==1.11.3', 'seaborn==0.13.0', 'scikit-lean==1.2.2',
        'tensorflow[and-cuda]==2.14.0'],
    scripts=['app/deepautocov'],
    zip_safe=False
)