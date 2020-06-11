from setuptools import setup, find_packages

setup(name='mot_neural_solver',
      packages=['mot_neural_solver',
                'mot_neural_solver.data',
                'mot_neural_solver.data.seq_processing',
                'mot_neural_solver.pl_module',
                'mot_neural_solver.models',
                'mot_neural_solver.tracker',
                'mot_neural_solver.utils'],
      package_dir={'':'src'},
      version='0.0.1',
      install_requires=[],)

