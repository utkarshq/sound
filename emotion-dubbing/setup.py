from setuptools import setup, find_packages

setup(
    name="emotion-dubbing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "opensmile>=2.4.0",
        "pydub>=0.25.0",
        "pyloudnorm>=0.1.1",
        "scikit-learn>=1.2.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0"
    ],
    python_requires=">=3.10",
)
