from setuptools import setup

setup(name='differentiation',
      version='0.1',
      description='calculate first dervative',
      url='-',
      author='Teemu Pukkila',
      author_email='teemu.pukkila@tuni.fi',
      license='--',
      packages=['first_derivate'],
      zip_safe=False,
      test_suite = 'nose.collector',
      tests_require = ['nose'])