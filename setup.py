from setuptools import setup, find_packages

setup(
    name='PBLES',
    version='0.0.2',
    author='Martin Kuhn',
    author_email='martin.kuhn@dfki.de',
    description='Private Bi-LSTM Event Log Synthesizer (PBLES)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/martinkuhn94/PBLES.git',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='Event Log Synthetization, Differential Privacy, Bi-LSTM, Synthetic Data Generation',
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.23.5',
        'tensorflow==2.14.0',
        'scipy==1.12.0',
        'keras==2.14.0',
        'pm4py==2.5.2',
        'scikit-learn==1.4.1.post1',
        'tensorflow_privacy==0.9.0',
        'openpyxl==3.1.2'
    ],
    python_requires='>=3.9',
)
