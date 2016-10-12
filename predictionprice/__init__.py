"""
Copyright (c) 2016 Tylor Darden
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import datetime
import smtplib
import email
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
import poloniex
import logging

class PredictionPrice(object):
    def __init__(self, currentPair="BTC_ETH", workingDirPath=".",
                 gmailAddress="", gmailAddressPassword="",
                 waitGettingTodaysChart = True, waitGettingTodaysChartTime = 60,
                 numFeature = 30, numTrainSample = 30, standarizationFeatureFlag = True, numStudyTrial = 50,
                 useBackTestOptResult=True, backTestInitialFund=1000, backTestSpread=0, backTestDays=60,
                 backTestOptNumFeatureMin=20, backTestOptNumFeatureMax=40, backTestOptNumTrainSampleMin=20, backTestOptNumTrainSampleMax=40):

        self.currentPair = currentPair
        self.workingDirPath = workingDirPath
        self.useBackTestOptResult=useBackTestOptResult
        if self.useBackTestOptResult and os.path.exists(self.workingDirPath + "/backTestOptResult_" + self.currentPair + ".pickle"):
            with open(self.workingDirPath + "/backTestOptResult_" + self.currentPair + ".pickle", mode='rb') as f:
                self.backTestOptResult_ = pickle.load(f)
            self.numFeature = self.backTestOptResult_["numFeatureOpt"]
            self.numTrainSample = self.backTestOptResult_["numTrainSampleOpt"]
        else:
            self.useBackTestOptResult = False
            self.numFeature = numFeature
            self.numTrainSample = numTrainSample
        self.standarizationFeatureFlag = standarizationFeatureFlag

        self.numStudyTrial = numStudyTrial
        self.gmailAddress = gmailAddress
        self.gmailAddressPassword = gmailAddressPassword

        self.waitGettingTodaysChart = waitGettingTodaysChart
        self.waitGettingTodaysChartTime = waitGettingTodaysChartTime

        self.backTestInitialFund = backTestInitialFund
        self.backTestSpread = backTestSpread
        self.backTestDays = backTestDays

        self.backTestOptNumFeatureMin = backTestOptNumFeatureMin
        self.backTestOptNumFeatureMax = backTestOptNumFeatureMax
        self.backTestOptNumTrainSampleMin = backTestOptNumTrainSampleMin
        self.backTestOptNumTrainSampleMax = backTestOptNumTrainSampleMax

        self.todayStr = str(datetime.datetime.today())[0:10]
        self.chartData_ = self.getChartData()
        #---self.saveChartData(self.chartData_)
        #---self.chartData_ = self.loadChartData()
        self.appreciationRate_ = self.getAppreciationRate(self.chartData_.open)
        self.chartDataLatestDayStr = str(self.chartData_.date[0])[0:10]

        if self.waitGettingTodaysChart:
            for tmpIndex in range(int(self.waitGettingTodaysChartTime*60.0/20.0)):
                if not (self.todayStr == self.chartDataLatestDayStr):
                    time.sleep(20)
                else:
                    break
                self.chartData_ = self.getChartData()
                self.appreciationRate_ = self.getAppreciationRate(self.chartData_.open)
                self.chartDataLatestDayStr = str(self.chartData_.date[0])[0:10]

    def sendMail(self,body):
        if self.gmailAddress=="" or self.gmailAddressPassword=="":
            return "Set your gmail address and password."
        # ---Create message
        msg = email.MIMEMultipart.MIMEMultipart()
        msg["From"] = self.gmailAddress
        msg["To"] = self.gmailAddress
        msg["Date"] = email.Utils.formatdate()
        msg["Subject"] = "TomorrowPricePrediction( " + self.currentPair + " )"
        msg.attach(email.MIMEText.MIMEText(body))
        # ---AttachimentFile
        attachimentFiles=[]
        if os.path.exists(self.workingDirPath + "/backTest_" + self.currentPair +".png"):
            attachimentFiles.append(self.workingDirPath + "/backTest_" + self.currentPair +".png")
        if os.path.exists(self.workingDirPath + "/backTestOptResult_" + self.currentPair +".png"):
            attachimentFiles.append(self.workingDirPath + "/backTestOptResult_" + self.currentPair + ".png")
        for afn in attachimentFiles:
            img = open(afn, "rb").read()
            mimg = email.MIMEImage.MIMEImage(img, "png", filename=afn)
            msg.attach(mimg)
        # ---SendMail
        smtpobj = smtplib.SMTP("smtp.gmail.com", 587)
        smtpobj.ehlo()
        smtpobj.starttls()
        smtpobj.login(self.gmailAddress, self.gmailAddressPassword)
        smtpobj.sendmail(self.gmailAddress, self.gmailAddress, msg.as_string())
        smtpobj.close()

    def fit(self, sampleData, classData):
        self.backTest(sampleData, classData, self.numFeature, self.numTrainSample, True)
        self.setTomorrowPriceProbability(sampleData, classData)

    def getComment(self):
        commentStr=""
        commentStr += "-----------------------------------------\n"
        commentStr += "Chart data info.\n"
        commentStr += "-----------------------------------------\n"
        commentStr += "CurrentPair: " + self.currentPair + "\n"
        commentStr += "Today: " + self.todayStr + "\n"
        commentStr += "LatestDayInData: " + self.chartDataLatestDayStr + "\n"
        commentStr += "LatestOpenPriceInData: " + str(self.chartData_.open[0]) + "\n"
        commentStr += "PreviousDayInData: " + str(self.chartData_.date[1])[0:10] + "\n"
        commentStr += "PreviousOpenPriceInData: " + str(self.chartData_.open[1]) + "\n"
        commentStr += "-----------------------------------------\n"
        commentStr += "Back test info.\n"
        commentStr += "-----------------------------------------\n"
        if self.useBackTestOptResult:
            commentStr += "ExecOptDay: " + str(self.backTestOptResult_["dateOpt"])[0:19] + "\n"
        else:
            commentStr += "ExecOptDay: Nan\n"
        commentStr += "NumFeature: " + str(self.numFeature) + "\n"
        commentStr += "NumTrainSample: " + str(self.numTrainSample) + "\n"
        commentStr += "AccuracyRateUp[%]: " + str(round(self.backTestResult_["AccuracyRateUp"].values[0]*100, 1)) + "\n"
        commentStr += "AccuracyRateDown[%]: " + str(round(self.backTestResult_["AccuracyRateDown"].values[0]*100, 1)) + "\n"
        commentStr += "InitialFund: " + str(self.backTestResult_["InitialFund"].values[0]) + "\n"
        commentStr += "FinalFund: " + str(self.backTestResult_["FinalFund"].values[0]) + "\n"
        commentStr += "IncreasedFundRatio[%]: " + str(round(self.backTestResult_["IncreasedFundRatio"].values[0]*100, 1)) + "\n"
        commentStr += "InitialCurrentPrice: " + str(self.backTestResult_["InitialCurrentPrice"].values[0]) + "\n"
        commentStr += "FinalCurrentPrice: " + str(self.backTestResult_["FinalCurrentPrice"].values[0]) + "\n"
        commentStr += "IncreasedCurrentPriceRatio[%]: " + str(round(self.backTestResult_["IncreasedCurrentPriceRatio"].values[0]*100, 1)) + "\n"
        commentStr += "-----------------------------------------\n"
        commentStr += "Tomorrow " + self.currentPair + " price prediction\n"
        commentStr += "-----------------------------------------\n"
        commentStr += "TomorrowPriceRise?: " + str(self.tomorrowPriceFlag_) +"\n"
        commentStr += "Probability[%]: " + str(round(self.tomorrowPriceProbability_*100,1)) +"\n"
        return commentStr


    def backTestOptimization(self, sampleData, classData):
        X = np.arange(self.backTestOptNumFeatureMin, self.backTestOptNumFeatureMax + 1, 1)
        Y = np.arange(self.backTestOptNumTrainSampleMin, self.backTestOptNumTrainSampleMax + 1, 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([len(Y[:]), len(X[0])])

        for i in range(0, len(X[0])):
            for j in range(0, len(Y[:])):
                Z[j][i] = self.backTest(sampleData, classData, X[j][i], Y[j][i], False)["IncreasedFundRatio"].values[0]
                print("-" * 80)
                print("NumFeatur: " + str(X[j][i]))
                print("NumTrainSample: " + str(Y[j][i]))
                print("IncreasedFundRatio[%]: " + str(round(Z[j][i] * 100, 1)))

        maxZRow = np.where(Z == np.max(Z))[0][0]
        maxZCol = np.where(Z == np.max(Z))[1][0]

        numFeatureOpt = X[maxZRow][maxZCol]
        numTrainSampleOpt = Y[maxZRow][maxZCol]
        dateOpt = datetime.datetime.now()

        backTestOptResult = {"X": X, "Y": Y, "Z": Z, "numFeatureOpt": numFeatureOpt,
                             "numTrainSampleOpt": numTrainSampleOpt, "dateOpt": dateOpt}
        with open(self.workingDirPath + "/backTestOptResult_" + self.currentPair + ".pickle", mode='wb') as f:
            pickle.dump(backTestOptResult, f)

        print("-" * 30 + " Optimization Result " + "-" * 30)
        print("NumFeatur: " + str(numFeatureOpt))
        print("NumTrainSample: " + str(numTrainSampleOpt))
        print("IncreasedFundRatio[%]: " + str(round(Z[maxZRow][maxZCol] * 100, 1)))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
        ax.contourf(X, Y, Z, zdir="z", offset=-2, cmap=plt.cm.hot)
        ax.set_title("Back test optimization (" + self.currentPair + ")")
        ax.set_xlabel("NumFeatur")
        ax.set_ylabel("NumTrainSample")
        ax.set_zlabel("IncreasedFundRatio")
        ax.view_init(90, 90)
        plt.savefig(self.workingDirPath + "/backTestOptResult_" + self.currentPair + ".png", dpi=50)
        plt.close()

    def backTest(self, sampleData, classData, numFeature, numTrainSample, saveBackTestGraph):
        Y = []
        YPrediction = []
        fund = [self.backTestInitialFund]
        pastDay = 0
        accuracyUp = 0
        accuracyDown = 0
        for trainStartIndex in range(self.backTestDays, 0, -1):
            yPrediction = self.quantizer(self.prediction(sampleData, classData, trainStartIndex, numFeature, numTrainSample))
            y = self.quantizer(classData[trainStartIndex - 1])
            Y.append(y.tolist())
            YPrediction.append(yPrediction.tolist())
            pastDay += 1
            if yPrediction == y:
                if yPrediction == 1:
                    accuracyUp += 1
                    fund.append(fund[pastDay - 1] * (
                    1 + abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                else:
                    accuracyDown += 1
                    fund.append(fund[pastDay - 1])
            else:
                if yPrediction == 1:
                    fund.append(fund[pastDay - 1] * (
                    1 - abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                else:
                    fund.append(fund[pastDay - 1])

        backTestAccuracyRateUp = float(accuracyUp) / sum(np.array(YPrediction)[np.where(np.array(YPrediction) == 1)])
        backTestAccuracyRateDown = -float(accuracyDown) / sum(np.array(YPrediction)[np.where(np.array(YPrediction) == -1)])

        trainStartIndex = 0
        backTestCurrentPrice = self.chartData_.open[trainStartIndex:trainStartIndex + self.backTestDays + 1]
        backTestCurrentPrice = backTestCurrentPrice[::-1].tolist()
        backTestDate = self.chartData_.date[trainStartIndex:trainStartIndex + self.backTestDays + 1]
        backTestDate = backTestDate[::-1].tolist()

        backTestFinalFund = fund[-1]
        backTestInitialCurrentPrice = backTestCurrentPrice[0]
        backTestFinalCurrentPrice = backTestCurrentPrice[-1]
        backTestIncreasedFundRatio = (backTestFinalFund - self.backTestInitialFund) / self.backTestInitialFund
        backTestIncreasedCurrentPriceRatio = (backTestFinalCurrentPrice - backTestInitialCurrentPrice) / backTestInitialCurrentPrice

        columnNames = ["AccuracyRateUp", "AccuracyRateDown",
                       "InitialFund", "FinalFund", "IncreasedFundRatio",
                       "InitialCurrentPrice", "FinalCurrentPrice", "IncreasedCurrentPriceRatio"]
        columnValues = [backTestAccuracyRateUp, backTestAccuracyRateDown,
                        self.backTestInitialFund, backTestFinalFund, backTestIncreasedFundRatio,
                        backTestInitialCurrentPrice, backTestFinalCurrentPrice, backTestIncreasedCurrentPriceRatio]
        backTestResult = pd.DataFrame(np.array([columnValues]), columns=columnNames)

        if saveBackTestGraph:
            fig1, ax1 = plt.subplots(figsize=(11, 6))
            p1, = ax1.plot(backTestDate, fund, "-ob")
            ax1.set_title("Back test (" + self.currentPair + ")")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Fund")
            plt.grid(fig1)
            ax2 = ax1.twinx()
            p2, = ax2.plot(backTestDate, backTestCurrentPrice, '-or')
            ax2.set_ylabel("Price[" + self.currentPair + "]")
            ax1.legend([p1, p2], ["Fund", "Price_" + self.currentPair], loc="upper left")
            plt.savefig(self.workingDirPath + "/backTest_" + self.currentPair + ".png", dpi=50)
            plt.close()

            self.backTestResult_ = backTestResult

        return backTestResult

    def setTomorrowPriceProbability(self, sampleData, classData):
        self.tomorrowPriceProbability_ = (self.prediction(sampleData, classData, 0, self.numFeature, self.numTrainSample) + 1.0) / 2.0
        if self.tomorrowPriceProbability_>0.5:
            self.tomorrowPriceFlag_ = True
        else:
            self.tomorrowPriceFlag_ = False
        return self.tomorrowPriceProbability_

    def prediction(self,sampleData,classData,trainStartIndex, numFeature, numTrainSample):
        train_X, train_y = self.preparationTrainSample(sampleData,classData,trainStartIndex, numFeature, numTrainSample)
        X = sampleData[trainStartIndex:trainStartIndex + numFeature]
        y = []
        for i in range(0, self.numStudyTrial):
            clf = tree.DecisionTreeClassifier()
            clf.fit(train_X, train_y)
            y.append(clf.predict([X])[0])
        return sum(y) * 1.0 / len(y)

    def standarizationFeature(self,X):
        numTrainSample, numFeature = np.shape(X)
        for i in range(numFeature):
            X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
        return X

    def quantizer(self, y):
        return np.where(np.array(y) >= 0.0, 1, -1)

    def preparationTrainSample(self,sampleData,classData,trainStartIndex, numFeature, numTrainSample):
        train_X = []
        train_y = []
        for i in range(numTrainSample):
            train_X.append(sampleData[trainStartIndex + i + 1:trainStartIndex + numFeature + i + 1])
            train_y.append(classData[trainStartIndex + i])
        if self.standarizationFeatureFlag:
            train_X = self.standarizationFeature(np.array(train_X))
        return np.array(train_X), np.array(train_y)

    def reverseDataFrame(self,dataFrame):
        dataFrame = dataFrame[::-1]
        dataFrame.index = dataFrame.index[::-1]
        return dataFrame

    def getChartData(self):
        polo = poloniex.Poloniex(timeout = 10, coach = True)
        chartData = pd.DataFrame(polo.marketChart(self.currentPair, period=polo.DAY, start=time.time() - polo.DAY * 500,end=time.time()))
        chartData.date = pd.DataFrame([datetime.datetime.fromtimestamp(chartData.date[i]).date() for i in range(len(chartData.date))])
        return self.reverseDataFrame(chartData)

    def saveChartData(self,chartData):
        with open("chartData_"+ self.currentPair + ".pickle", mode="wb") as f:
            pickle.dump(chartData, f)
        return

    def loadChartData(self):
        with open("chartData_"+ self.currentPair + ".pickle", mode="rb") as f:
            chartData = pickle.load(f)
        return chartData

    def getAppreciationRate(self,price):
        return np.append(-np.diff(price) / price[1:].values,0)



class CustumPoloniex(poloniex.Poloniex):
    def __init__(self, APIKey=False, Secret=False,timeout=10, coach=True, loglevel=logging.WARNING, basicCoin="BTC",
                 workingDirPath=".", gmailAddress="", gmailAddressPassword="",
                 coins=[], buySigns=[] ):
        super(CustumPoloniex, self).__init__(APIKey, Secret, timeout, coach, loglevel)
        self.basicCoin = basicCoin
        self.workingDirPath = workingDirPath
        self.gmailAddress = gmailAddress
        self.gmailAddressPassword = gmailAddressPassword
        self.coins = coins
        self.buySigns = buySigns

        
    def myAvailableCompleteBalances(self):
        """return AvailableCompleteBalances as pandas.DataFrame."""
        balance = pd.DataFrame.from_dict(self.myCompleteBalances()).T
        return balance.iloc[np.where(balance["btcValue"] != "0.00000000")]
    
    def myEstimatedValueOfHoldings(self):
        """return EstimatedValueOfHoldings."""
        balance = self.myAvailableCompleteBalances()
        estimatedValueOfHoldingsAsBTC = np.sum(np.float_(balance.iloc[:, 1]))
        lastValueUSDT_BTC = pd.DataFrame.from_dict(self.marketTicker()).T.loc["USDT_BTC"]["last"]
        estimatedValueOfHoldingsAsUSD = np.float_(lastValueUSDT_BTC) * estimatedValueOfHoldingsAsBTC
        return estimatedValueOfHoldingsAsBTC, estimatedValueOfHoldingsAsUSD
    
    def cancelOnOrder(self,coin):
        """cancel on Order"""
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==coin)[0])==0: return
        
        while balance.loc[coin]["onOrders"]!="0.00000000":
            orderId=pd.DataFrame.from_dict(self.myOrders(self.basicCoin + "_" + coin))["orderNumber"].tolist()[0]
            self.cancelOrder(orderId)
            balance = self.myAvailableCompleteBalances()
            
    def marketSellAll(self, coin):
        """Sell all coin with market price."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==coin)[0])==0: return
        bids=pd.DataFrame.from_dict(self.marketOrders(self.basicCoin + "_" + coin)).bids
        sumBtcValue=0.0
        for rate, amount in zip(np.array(bids.tolist())[:,0],np.array(bids.tolist())[:,1]):
            sumBtcValue += float(rate)*float(amount)
            if float(balance.loc[coin]["btcValue"]) < sumBtcValue:
                break
        return self.sell(self.basicCoin + "_" + coin, rate, balance.loc[coin]["available"])

    def marketSell(self, coin, btcValue):
        """Sell coin with market price as estimated btcValue."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==coin)[0])==0: return
        if float(btcValue)>float(balance.loc[coin]["btcValue"]):
            return self.marketSellAll(coin)
        bids = pd.DataFrame.from_dict(self.marketOrders(self.basicCoin + "_" + coin)).bids
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(bids.tolist())[:,0],np.array(bids.tolist())[:,1]):
            oldSumAmount = sumAmount
            sumAmount += float(amount)
            oldSumBtcValue = sumBtcValue
            sumBtcValue += float(rate)*float(amount)
            if float(btcValue) < sumBtcValue:
                break
        coinAmount = np.floor((oldSumAmount + (float(btcValue) - float(oldSumBtcValue))/float(rate)) * 1e8) * 1e-8
        return self.sell(self.basicCoin + "_" + coin, rate, coinAmount)
    
    
    def marketBuyAll(self, coin):
        """Buy coin with market price as much as possible."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index=="BTC")[0])==0: return
        asks = pd.DataFrame.from_dict(self.marketOrders(self.basicCoin + "_" + coin)).asks
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(asks.tolist())[:,0],np.array(asks.tolist())[:,1]):
            oldSumAmount = sumAmount
            sumAmount += float(amount)
            oldSumBtcValue = sumBtcValue
            sumBtcValue += float(rate)*float(amount)
            if float(balance.loc["BTC"]["btcValue"]) < sumBtcValue:
                break
        coinAmount = np.floor((oldSumAmount + (float(balance.loc["BTC"]["btcValue"]) - float(oldSumBtcValue)) / float(rate)) * 1e8) * 1e-8
        if float(rate)*coinAmount<0.0001:
            return
        return self.buy(self.basicCoin + "_" + coin, rate, coinAmount)
    
    def marketBuy(self, coin, btcValue):
        """Buy coin with market price as estimated btcValue."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index=="BTC")[0])==0: return
        if float(btcValue)>float(balance.loc["BTC"]["btcValue"]):
            return self.marketBuyAll(coin)
        asks = pd.DataFrame.from_dict(self.marketOrders(self.basicCoin + "_" + coin)).asks
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(asks.tolist())[:,0],np.array(asks.tolist())[:,1]):
            oldSumAmount = sumAmount
            sumAmount += float(amount)
            oldSumBtcValue = sumBtcValue
            sumBtcValue += float(rate)*float(amount)
            if float(btcValue) < sumBtcValue:
                break
        coinAmount = np.floor((oldSumAmount + (float(btcValue) - float(oldSumBtcValue)) / float(rate)) * 1e8) * 1e-8
        return self.buy(self.basicCoin + "_" + coin, rate, coinAmount)

    def fitSell(self):
        """Sell coins in accordance with buySigns."""
        balance=self.myAvailableCompleteBalances()  
        for coinIndex in range(len(self.coins)):
            if not self.buySigns[coinIndex]: #Sign is Sell?
                if not (len(np.where(balance.index==self.coins[coinIndex])[0])==0): #Holding the coin?
                    self.marketSellAll(self.coins[coinIndex])
                    
    def fitBuy(self):
        """Buy coins in accordance with buySigns."""
        balance=self.myAvailableCompleteBalances()
        if np.sum(self.buySigns)==0: # All signs are sell?
            return
        else:
            myBTC,myUSD = self.myEstimatedValueOfHoldings()
            distributionBTCValue = myBTC*1.0/np.sum(self.buySigns)

            for coinIndex in range(len(self.coins)):
                if self.buySigns[coinIndex]: #Sign is Buy?
                    if not (len(np.where(balance.index==self.coins[coinIndex])[0])==0): #Holding this coin?
                        extraBTCValue=float(balance.loc[self.coins[coinIndex]]["btcValue"])-float(distributionBTCValue)
                        if extraBTCValue>0:
                            self.marketSell(self.coins[coinIndex],extraBTCValue)
                        else:
                            self.marketBuy(self.coins[coinIndex],np.abs(extraBTCValue))
                    else:
                        self.marketBuy(self.coins[coinIndex],distributionBTCValue)
                        
    def fitBalance(self):
        """Call fitSell and fitBuy."""
        self.fitSell()
        self.fitBuy()
        
    def sendMailBalance(self):
        """Send the balance by e-mail."""
        if self.gmailAddress=="" or self.gmailAddressPassword=="":
            return "Set your gmail address and password."
        # ---Create message
        myBTC,myUSD = self.myEstimatedValueOfHoldings()
        balance = self.myAvailableCompleteBalances()
        body = ""
        body += "Coins: " + str(self.coins) + "\n"
        body += "BuySigns: " + np.str(self.buySigns) + "\n"
        body += "\n"
        body += "Your total fund:\n"
        body +=  str(myBTC) + " BTC\n"
        body +=  str(myUSD) + " USD\n"
        body += "\n"
        body += "Breakdown:\n"
        body += str(balance)
        msg = email.MIMEMultipart.MIMEMultipart()
        msg["From"] = self.gmailAddress
        msg["To"] = self.gmailAddress
        msg["Date"] = email.Utils.formatdate()
        msg["Subject"] = "Poloniex Balance"
        msg.attach(email.MIMEText.MIMEText(body))
        # ---SendMail
        smtpobj = smtplib.SMTP("smtp.gmail.com", 587)
        smtpobj.ehlo()
        smtpobj.starttls()
        smtpobj.login(self.gmailAddress, self.gmailAddressPassword)
        smtpobj.sendmail(self.gmailAddress, self.gmailAddress, msg.as_string())
        smtpobj.close()

    def savePoloniexBalanceToCsv(self):
        """Save EstimatedValueOfHoldings to csv file."""
        fileName = self.workingDirPath + "/PoloniexBalance.csv"
        date = str(datetime.datetime.today())[0:19]
        myBTC, myUSD = self.myEstimatedValueOfHoldings()
        if os.path.exists(fileName):
            f = open(fileName, "a")
            writer = csv.writer(f, lineterminator="\n")
        else:
            f = open(fileName, "w")
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Date","BTC","USD"])  # Write header
        writer.writerow([date,str(myBTC),str(myUSD)])
        f.close()
