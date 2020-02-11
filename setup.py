import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='statlab',
     version='0.7',
     scripts=['statlab_doc'] ,
     author="StVladislav",
     author_email="sv6382@gmail.com",
     description="Statlab package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/StVladislav/statlab",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
