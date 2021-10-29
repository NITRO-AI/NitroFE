from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(name='NitroFE',
      version='0.0.7',
      url='https://github.com/NITRO-AI/NitroFE',
      author='Nitro-AI',
      author_email='nitro.ai.solutions@gmail.com',
      license='Apache License 2.0',
      packages=['NitroFE','NitroFE.encoding.','NitroFE.time_based_features','NitroFE.time_based_features.indicator_features','NitroFE.time_based_features.moving_average_features','NitroFE.time_based_features.weighted_window_features'],
      zip_safe=True,
	description="NitroFE is a Python feature engineering engine which provides a variety of feature engineering modules designed to handle continous calcualtion.",
      long_description=long_description  ,
	long_description_content_type='text/markdown',
	  install_requires=[
		"pandas",
		"numpy",
		"scipy",
            "plotly"
		],
      )