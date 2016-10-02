# -*- coding: utf-8 -*-
from predictionprice import PredictionPrice


myGmailAddress = "*******@gmail.com"
myGmailAddressPassword = "*******"
workingDirPath="."
basicCoin="BTC"
coins=["ETH","XMR","XRP","FCT","DASH"]
backTestOptParams=[
    [20, 40, 20, 40],
    [20, 40, 20, 40],
    [20, 40, 20, 40],
    [20, 40, 20, 40],
    [20, 40, 20, 40]]
ppList=[]
tommorrwPricePrediction=[]

#---Prediction price and back test
for coinIndex in range(0,1): #---range(len(coins)):
    pp = PredictionPrice(currentPair=basicCoin+"_"+coins[coinIndex], workingDirPath=workingDirPath,
                         gmailAddress=myGmailAddress, gmailAddressPassword=myGmailAddressPassword,
                         backTestOptNumFeatureMin=backTestOptParams[coinIndex][0], backTestOptNumFeatureMax=backTestOptParams[coinIndex][1],
                         backTestOptNumTrainSampleMin=backTestOptParams[coinIndex][2], backTestOptNumTrainSampleMax=backTestOptParams[coinIndex][3])

    pp.fit(pp.appreciationRate_,pp.quantizer(pp.appreciationRate_))
    pp.sendMail(pp.getComment())
    ppList.append(pp)
    tommorrwPricePrediction.append(pp.tommorrowPriceFlag_)

#--- back test optimization
for coinIndex in range(len(coins)):
    pp = ppList[coinIndex]
    pp.backTestOptimization(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))




