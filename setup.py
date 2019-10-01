from distutils.core import setup
setup(
  name = 'pyGRNN',         # How you named your package folder (MyLib)
  packages = ['pyGRNN'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Python implementation of General Regression Neural Network (Nadaraya-Watson Estimator). A Feature Selection module based on GRNN is also provided',   # Give a short description about your library
  author = 'Federico Amato',                   # Type in your name
  author_email = 'federico.amato@unil.ch',      # Type in your E-Mail
  url = 'https://github.com/federhub',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/federhub/pyGRNN/archive/v0.0.1.tar.gz',    # I explain this later on
  keywords = ['Machine Learning', 'General Regression Neural Network', 'Kernel Regression', 'Feature Selection'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
          'pandas',
          'numpy',
          'seaborn',
          'sklearn',
          'itertools',
          'matplotlib',
          'scipy',
          'operator',
          'itertools' 
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)