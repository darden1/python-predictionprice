import os
import smtplib
import email
import csv
import pytz
import time
import datetime
import logging
import numpy as np
import pandas as pd
import poloniex


class ExchangeTradePoloniex(poloniex.Poloniex):
    def __init__(self, APIKey=False, Secret=False,timeout=10, coach=True, loglevel=logging.WARNING, extend=True, basicCoin="BTC",
                 workingDirPath=".", gmailAddress="", gmailAddressPassword="",
                 coins=[], buySigns=[] ):
        super(ExchangeTradePoloniex, self).__init__(APIKey, Secret, timeout, coach, loglevel, extend)
        self.basicCoin = basicCoin
        self.workingDirPath = workingDirPath
        self.gmailAddress = gmailAddress
        self.gmailAddressPassword = gmailAddressPassword
        self.coins = coins
        self.buySigns = buySigns
        self.todayStr = str(datetime.datetime.now(pytz.timezone("UTC")))[0:10]

    def myAvailableCompleteBalances(self):
        """Return AvailableCompleteBalances as pandas.DataFrame."""
        balance = pd.DataFrame.from_dict(self.myCompleteBalances(account="exchange")).T
        return balance.iloc[np.where(balance["btcValue"] != "0.00000000")]

    def myEstimatedValueOfHoldings(self):
        """Return EstimatedValueOfHoldings."""
        balance = self.myAvailableCompleteBalances()
        estimatedValueOfHoldingsAsBTC = np.sum(np.float_(balance.iloc[:, 1]))
        lastValueUSDT_BTC = pd.DataFrame.from_dict(self.marketTicker()).T.loc["USDT_BTC"]["last"]
        estimatedValueOfHoldingsAsUSD = np.float_(lastValueUSDT_BTC) * estimatedValueOfHoldingsAsBTC
        return estimatedValueOfHoldingsAsBTC, estimatedValueOfHoldingsAsUSD

    def cancelOnOrder(self,coin):
        """Cancel on Exchange Order"""
        onOrders = pd.DataFrame.from_dict(self.myOrders(pair=self.basicCoin + "_" + coin))
        if len(onOrders) == 0: return
        onExchangeOrders = onOrders.loc[np.where(onOrders["margin"]==0)]
        if len(onExchangeOrders) == 0: return
    
        while len(onExchangeOrders) != 0:
            orderId = onExchangeOrders["orderNumber"].tolist()[0]
            self.cancelOrder(orderId)
            onOrders = pd.DataFrame.from_dict(self.myOrders(pair=self.basicCoin + "_" + coin))
            if len(onOrders) == 0: 
                return
            else:
                onExchangeOrders = onOrders.loc[np.where(onOrders["margin"]==0)]

    def marketSell(self, coin, btcValue):
        """Sell coin with market price as estimated btcValue."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==coin)[0])==0: return
        if float(btcValue)>float(balance.loc[coin]["btcValue"]):
            return self.marketSellAll(coin)
        bids = pd.Series(pd.DataFrame.from_dict(self.marketOrders(pair=self.basicCoin + "_" + coin, depth=1000)["bids"]).values.tolist())
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(bids.tolist())[:,0],np.array(bids.tolist())[:,1]):
            sumAmount += float(amount)
            sumBtcValue += float(rate)*float(amount)
            if float(btcValue) < sumBtcValue:
                break
        coinAmount = np.floor((sumAmount - (float(sumBtcValue) - float(btcValue))/float(rate)) * 1e7) * 1e-7
        return self.sell(self.basicCoin + "_" + coin, rate, coinAmount)

    def marketSellAll(self, coin):
        """Sell all coin with market price."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==coin)[0])==0: return
        bids = pd.Series(pd.DataFrame.from_dict(self.marketOrders(pair=self.basicCoin + "_" + coin, depth=1000)["bids"]).values.tolist())
        sumBtcValue = 0.0
        for rate, amount in zip(np.array(bids.tolist())[:,0],np.array(bids.tolist())[:,1]):
            sumBtcValue += float(rate)*float(amount)
            if float(balance.loc[coin]["btcValue"]) < sumBtcValue:
                break
        return self.sell(self.basicCoin + "_" + coin, rate, balance.loc[coin]["available"])

    def marketBuy(self, coin, btcValue):
        """Buy coin with market price as estimated btcValue."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index==self.basicCoin)[0])==0: return
        if float(btcValue)>float(balance.loc[self.basicCoin]["btcValue"]):
            return self.marketBuyAll(coin)
        asks = pd.Series(pd.DataFrame.from_dict(self.marketOrders(pair=self.basicCoin + "_" + coin, depth=1000)["asks"]).values.tolist())
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(asks.tolist())[:,0],np.array(asks.tolist())[:,1]):
            sumAmount += float(amount)
            sumBtcValue += float(rate)*float(amount)
            if float(btcValue) < sumBtcValue:
                break
        coinAmount = np.floor((sumAmount - (float(sumBtcValue) - float(btcValue)) / float(rate)) * 1e7) * 1e-7
        return self.buy(self.basicCoin + "_" + coin, rate, coinAmount)

    def marketBuyAll(self, coin):
        """Buy coin with market price as much as possible."""
        self.cancelOnOrder(coin)
        balance = self.myAvailableCompleteBalances()
        if len(np.where(balance.index == self.basicCoin)[0]) == 0: return
        asks = pd.Series(pd.DataFrame.from_dict(self.marketOrders(pair=self.basicCoin + "_" + coin, depth=1000)["asks"]).values.tolist())
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(asks.tolist())[:, 0], np.array(asks.tolist())[:, 1]):
            sumAmount += float(amount)
            sumBtcValue += float(rate) * float(amount)
            if float(balance.loc[self.basicCoin]["btcValue"]) < sumBtcValue:
                break
        coinAmount = np.floor((
                              sumAmount - (float(sumBtcValue) - float(balance.loc[self.basicCoin]["btcValue"])) / float(
                                  rate)) * 1e7) * 1e-7
        if float(rate) * coinAmount < 0.0001:
            return
        return self.buy(self.basicCoin + "_" + coin, rate, coinAmount)
    
    def fitSell(self):
        """Sell coins in accordance with buySigns."""
        balance = self.myAvailableCompleteBalances()
        for coinIndex in range(len(self.coins)):
            if not self.buySigns[coinIndex]: #Sign is Sell?
                if len(np.where(balance.index == self.coins[coinIndex])[0]) != 0:  # Holding the coin?
                    self.marketSellAll(self.coins[coinIndex])

    def fitBuy(self):
        """Buy coins in accordance with buySigns."""
        balance = self.myAvailableCompleteBalances()
        if np.sum(self.buySigns)==0: # All signs are sell?
            return
        else:
            myBTC,myUSD = self.myEstimatedValueOfHoldings()
            distributionBTCValue = myBTC*1.0/np.sum(self.buySigns)
            # --- Sell extra coins
            numSelledExtraCoins = 0
            for coinIndex in range(len(self.coins)):
                if self.buySigns[coinIndex]: #Sign is Buy?
                    if len(np.where(balance.index == self.coins[coinIndex])[0]) != 0:  # Holding the coin?
                        extraBTCValue = float(balance.loc[self.coins[coinIndex]]["btcValue"]) - float(distributionBTCValue)
                        if extraBTCValue>0:
                            self.marketSell(self.coins[coinIndex],extraBTCValue)
                            numSelledExtraCoins += 1

            # --- Buy coins by distlibuted btcValue
            balance = self.myAvailableCompleteBalances()
            myBTC,myUSD = self.myEstimatedValueOfHoldings()
            distributionBTCValue = myBTC*1.0/np.sum(self.buySigns)
            for coinIndex in range(len(self.coins)):
                if self.buySigns[coinIndex]:  # Sign is Buy?
                    if len(np.where(balance.index == self.coins[coinIndex])[0]) != 0:  # Holding the coin?
                        extraBTCValue = float(balance.loc[self.coins[coinIndex]]["btcValue"]) - float(distributionBTCValue)
                        if extraBTCValue < 0:
                            self.marketBuy(self.coins[coinIndex], np.abs(extraBTCValue))
                    else:
                        self.marketBuy(self.coins[coinIndex], distributionBTCValue)

    def fitBalance(self):
        """Call fitSell and fitBuy."""
        self.fitSell()
        self.fitBuy()

    def getSummary(self):
        myBTC, myUSD = self.myEstimatedValueOfHoldings()
        balance = self.myAvailableCompleteBalances()
        summaryStr = ""
        summaryStr += "-----------------------------------------\n"
        summaryStr += "Poloniex Balance.\n"
        summaryStr += "-----------------------------------------\n"
        summaryStr += "Today: " + self.todayStr + "\n"
        summaryStr += "Coins: " + str(self.coins) + "\n"
        summaryStr += "BuySigns: " + np.str(self.buySigns) + "\n"
        summaryStr += "\n"
        summaryStr += "Your total fund in exchange account:\n"
        summaryStr += str(myBTC) + " BTC\n"
        summaryStr += str(myUSD) + " USD\n"
        summaryStr += "\n"
        summaryStr += "Breakdown:\n"
        summaryStr += str(balance)
        return summaryStr

    def sendMailBalance(self, body):
        """Send the balance by e-mail."""
        if self.gmailAddress == "" or self.gmailAddressPassword == "":
            return "Set your gmail address and password."
        # ---Create message
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
