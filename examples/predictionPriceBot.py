# -*- coding: utf-8 -*-
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from predictionprice import CustumPoloniex
from predictionprice import PredictionPrice


def botRoutine():
    myGmailAddress = "********@gmail.com"
    myGmailAddressPassword = "************"
    myAPIKey="************************"
    mySecret="************************************************"

    coins = ["ETH", "XMR", "XRP", "FCT", "DASH"]
    backTestOptParams = [
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40]]

    basicCoin = "BTC"
    workingDirPath = os.path.dirname(os.path.abspath(__file__))

    ppList = []
    tomorrwPricePrediction = []

    # --- Prediction price and back test
    for coinIndex in range(len(coins)):
        pp = PredictionPrice(currentPair = basicCoin + "_" + coins[coinIndex], workingDirPath = workingDirPath,
                             gmailAddress = myGmailAddress, gmailAddressPassword = myGmailAddressPassword,
                             backTestOptNumFeatureMin = backTestOptParams[coinIndex][0],
                             backTestOptNumFeatureMax = backTestOptParams[coinIndex][1],
                             backTestOptNumTrainSampleMin = backTestOptParams[coinIndex][2],
                             backTestOptNumTrainSampleMax = backTestOptParams[coinIndex][3])

        pp.fit(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))
        pp.sendMail(pp.getComment())
        ppList.append(pp)
        tomorrwPricePrediction.append(pp.tomorrowPriceFlag_)

    # --- Fit balance
    polo = CustumPoloniex(APIKey = myAPIKey, Secret = mySecret, workingDirPath = workingDirPath,
                          gmailAddress = myGmailAddress, gmailAddressPassword = myGmailAddressPassword,
                          coins = coins, buySigns = tomorrwPricePrediction)
    polo.fitBalance()
    polo.sendMailBalance()
    polo.savePoloniexBalanceToCsv()

    # --- Back test optimization
    for coinIndex in range(len(coins)):
        pp = ppList[coinIndex]
        pp.backTestOptimization(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))


if __name__ == "__main__":
    sc = BlockingScheduler(timezone="UTC")
    sc.add_job(botRoutine, "cron", hour=0, minute=1)
    sc.start()
