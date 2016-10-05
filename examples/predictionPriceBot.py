# -*- coding: utf-8 -*-
from predictionprice import PredictionPrice
from predictionprice import CustumPoloniex
from apscheduler.schedulers.gevent import GeventScheduler


def botRoutine():
    myGmailAddress = "********@gmail.com"
    myGmailAddressPassword = "************"
    myAPIKey="************************"
    mySecret="************************************************"
    
    basicCoin="BTC"
    workingDirPath="."
    
    coins = ["ETH", "XMR", "XRP", "FCT", "DASH"]
    backTestOptParams = [
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40],
        [20, 40, 20, 40]]
    ppList = []
    tommorrwPricePrediction = []
    # ---Prediction price and back test
    for coinIndex in range(len(coins)):
        pp = PredictionPrice(currentPair = basicCoin + "_" + coins[coinIndex], workingDirPath = workingDirPath,
                             waitGettingTodaysChart=False,
                             gmailAddress = myGmailAddress, gmailAddressPassword = myGmailAddressPassword,
                             backTestOptNumFeatureMin = backTestOptParams[coinIndex][0],
                             backTestOptNumFeatureMax = backTestOptParams[coinIndex][1],
                             backTestOptNumTrainSampleMin = backTestOptParams[coinIndex][2],
                             backTestOptNumTrainSampleMax = backTestOptParams[coinIndex][3])

        pp.fit(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))
        pp.sendMail(pp.getComment())
        ppList.append(pp)
        tommorrwPricePrediction.append(pp.tommorrowPriceFlag_)

    # --- Fit balance
    polo = CustumPoloniex(APIKey = myAPIKey, Secret = mySecret, workingDirPath = workingDirPath,
                          gmailAddress = myGmailAddress, gmailAddressPassword = myGmailAddressPassword)
    polo.fitBalance(coins, tommorrwPricePrediction)
    polo.sendMailBalance()
    polo.savePoloniexBalanceToCsv()

    # --- Back test optimization
    for coinIndex in range(len(coins)):
        pp = ppList[coinIndex]
        pp.backTestOptimization(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))


if __name__ == "__main__":
    scheduler = GeventScheduler()
    scheduler.add_job(botRoutine, "cron", hour=9, minute=5)
    g = scheduler.start()  # g is the greenlet that runs the scheduler loop
    g.join()




