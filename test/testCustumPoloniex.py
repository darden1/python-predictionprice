# -*- coding: utf-8 -*-
from predictionprice import CustumPoloniex

myAPIKey = "************"
mySecret = "**********************************************************"

polo = CustumPoloniex(APIKey=myAPIKey, Secret=mySecret, timeout=10, coach=True)
myBTC, myUSD = polo.myEstimatedValueOfHoldings()
balance = polo.myAvailableCompleteBalances()

print("-"*35)
print("Your total fund:")
print(str(myBTC) + " BTC")
print(str(myUSD) + " USD")
print("\nBreakdown:")
print(balance)
