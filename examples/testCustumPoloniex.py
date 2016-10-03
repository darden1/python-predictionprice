# -*- coding: utf-8 -*-
from predictionprice import CustumPoloniex

myAPIKey="************"
mySecret="**********************************************************"

polo = CustumPoloniex(APIKey=myAPIKey, Secret=mySecret, timeout=3, coach=True)
balance=polo.myAvailableCompleteBalances()

myBTC,myUSD=polo.myEstimatedValueOfHoldings()

print("Your Fund is " + str(myBTC) + " BTC")
print("Your Fund is " + str(myUSD) + " USD")
balance