from setuptools import setup

setup(name='num_calculus',
      version='0.1',
      description='Numerical calculus tools',
      url='https://gitlab.com/saarioka/compphys',
      author='Santeri Saariokari',
      author_email='santeri.saariokari@tuni.fi',
      license='MIT',
      packages=['num_calculus'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
