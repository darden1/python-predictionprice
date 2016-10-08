from setuptools import setup, find_packages
 
setup(
        name             = "predictionprice",
        version          = "1.0.1",
        description      = "Prediction price for Python 2.7.x",
        license          = "MIT",
        author           = "darden1",
        author_email     = "darden066@gmail.com",
        url              = "https://github.com/darden1/python-predictionprice/",
        keywords         = "",
        packages         = find_packages(),
        install_requires = ["numpy","pandas","matplotlib","scikit-learn","apscheduler","poloniex"],
        dependency_links = ["https://codeload.github.com/s4w3d0ff/python-poloniex/zip/master#egg=poloniex"],
        zip_safe=False
        )