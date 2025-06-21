from setuptools import find_packages
from distutils.core import setup

setup(name='tft_legged_gym',
      version='1.0.0',
      author='Tommy Tang',
      packages=find_packages(),
      author_email='feitong_Tang@163.com',
      description='personal project for legged robot control',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
