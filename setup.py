from setuptools import setup, find_packages

setup(
    name="taixiu-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.3.0',
        'xgboost>=1.7.6', 
        'lightgbm>=4.1.0',
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
    ],
    author="Your Name",
    description="AI System for Tai Xiu Prediction",
    python_requires='>=3.8',
)