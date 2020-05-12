import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='statlab',
     version='0.9.1',
     scripts=['statlab_doc'] ,
     author="St.Vladislav",
     author_email="sv6382@gmail.com",
     description="Statlab package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/SLprojects/statlab",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
