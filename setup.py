from setuptools import setup
import versioneer

setup(name='simplex',
      description='Tools for building core set MSMs in TICA space',
      url='https://github.com/fabian-paul/simplex',
      author='Fabian Paul',
      author_email='fab@zedat.fu-berlin.de',
      license='LGPLv3+',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=['simplex'],
      zip_safe=False)