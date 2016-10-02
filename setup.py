from setuptools import setup, find_packages
 
setup(
        name             = "predictionprice",
        version          = "1.0.0",
        description      = "Prediction price for Python 2.7.x",
        license          = "MIT",
        author           = "darden1",
        author_email     = "darden066@gmail.com",
        url              = "https://github.com/",
        keywords         = "",
        packages         = find_packages(),
        install_requires = ["numpy","pandas","matplotlib","sklearn"],
        zip_safe=False
        )
