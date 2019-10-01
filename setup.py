from distutils.core import setup
setup(
  name = 'pyGRNN',         
  packages = ['pyGRNN'],   
  version = 'v0.0.2.1',      
  license='MIT',  
  README = (./"README.md").read_text(),
  description = 'Python implementation of General Regression Neural Network (Nadaraya-Watson Estimator). A Feature Selection module based on GRNN is also provided',   # Give a short description about your library
  author = 'Federico Amato',                   
  author_email = 'federico.amato@unil.ch',      #
  url = 'https://github.com/federhub/pyGRNN',  
  download_url = 'https://github.com/federhub/pyGRNN/archive/v0.0.2.1.tar.gz',    
  keywords = ['Machine Learning', 'General Regression Neural Network', 'Kernel Regression', 'Feature Selection'],   
  install_requires=[
          'pandas',
          'numpy',
          'seaborn',
          'scikit-learn',
          'matplotlib',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',     
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.6',
  ],
)
