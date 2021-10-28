from setuptools import setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(name='NitroFE',
      version='0.0.3',
      url='https://github.com/NITRO-AI/NitroFE',
      license='Apache License 2.0',
      packages=['NitroFE'],
      zip_safe=True,
	description="NitroFE is a Python feature engineering engine which provides a variety of feature engineering modules designed to handle continous calcualtion.",
      long_description=long_description  ,
	long_description_content_type='text/markdown',
	  install_requires=[
		"pandas",
		"numpy",
		"scipy",
        "plotly"
		])