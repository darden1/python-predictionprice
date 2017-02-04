from setuptools import setup, find_packages
 
setup(
        name             = "predictionprice",
        version          = "2.0.0",
        description      = "Prediction price for Python 2.7.x",
        license          = "MIT",
        author           = "darden1",
        author_email     = "darden066@gmail.com",
        url              = "https://github.com/darden1/python-predictionprice/",
        keywords         = "",
        packages         = find_packages(),
        install_requires = ["numpy","pandas","matplotlib","scikit-learn","apscheduler","poloniex==0.2.2"],
        dependency_links = ["git+https://git@github.com/darden1/python-poloniex.git@master#egg=poloniex-0.2.2"],
        zip_safe=False
        )