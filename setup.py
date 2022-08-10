import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='reinforcement-learning',
    version='0.0.1',
    author='J. Spieler',
    author_email='jonathanspi30@gmail.com',
    description='Deep Reinforcement Learning Algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jspieler/reinforcement-learning',
    project_urls = {
        "Bug Tracker": "https://github.com/jspieler/reinforcement-learning/issues"
    },
    license='MIT',
    packages=['PyTorch', 'TensorFlow2'],
    install_requires=['numpy', 'torch', 'torchvision', 'torchaudio', 'tensorflow', 'matplotlib', 'gym'],
)