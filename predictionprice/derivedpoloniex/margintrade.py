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


class MarginTradePoloniex(poloniex.Poloniex):
    def __init__(self, Key=False, Secret=False,timeout=10, coach=True, loglevel=logging.WARNING, extend=True, basicCoin="BTC",
                 workingDirPath=".", gmailAddress="", gmailAddressPassword="",
                 coins=[], tradeSigns=[] ):
        super(MarginTradePoloniex, self).__init__(Key, Secret, timeout, coach, loglevel, extend)
        self.basicCoin = basicCoin
        self.workingDirPath = workingDirPath
        self.gmailAddress = gmailAddress
        self.gmailAddressPassword = gmailAddressPassword
        self.coins = coins
        self.tradeSigns = tradeSigns
        self.todayStr = str(datetime.datetime.now(pytz.timezone("UTC")))[0:10]
        self.leverage = 2.5


    def floatToEighthDigit(self, numFloat):
        """Change float number to string of eighth digit number."""
        return "{0:.9f}".format(float(numFloat)).split(".")[0] + "." + "{0:.9f}".format(float(numFloat)).split(".")[1][0:8]

    def returnSummary(self):
        """Return margin account balance summary as pandas data frame."""
        return pd.DataFrame.from_dict({"summary": self.returnMarginAccountSummary()})

    def returnTradableBalance(self):
        """Return tradable balance."""
        summary = self.returnSummary()
        return self.floatToEighthDigit(np.float(summary.loc["netValue"]) * self.leverage - np.float(summary.loc["totalBorrowedValue"]))

    def getOpeningMarginPosition(self):
        """Return AvailableCompleteBalances as pandas.DataFrame."""
        position = pd.DataFrame.from_dict(self.getMarginPosition()).T
        if len(np.where(position["amount"] != "0.00000000")[0]) == 0:
            return False
        else:
            return position.iloc[np.where(position["amount"] != "0.00000000")]

    def cancelOnMarginOrder(self, coin):
        """Cancel on Margin Order"""
        onOrders = pd.DataFrame.from_dict(self.returnOpenOrders(pair=self.basicCoin + "_" + coin))
        if len(onOrders) == 0:
            return
        onMarginOrders = onOrders.loc[np.where(onOrders["margin"] == 1)]
        if len(onMarginOrders) == 0:
            return
        while len(onMarginOrders) != 0:
            orderId = onMarginOrders["orderNumber"].tolist()[0]
            self.cancelOrder(orderId)
            onOrders = pd.DataFrame.from_dict(self.returnOpenOrders(pair=self.basicCoin + "_" + coin))
            if len(onOrders) == 0:
                return
            else:
                onMarginOrders = onOrders.loc[np.where(onOrders["margin"] == 1)]

    def returnRateAndAmount(self, orderStr, coin, btcValue):
        """Return BTC rate and coin amount to trade some coin."""
        order = pd.Series(pd.DataFrame.from_dict(
            self.marketOrders(pair=self.basicCoin + "_" + coin, depth=1000)[orderStr]).values.tolist())
        sumBtcValue = 0.0
        sumAmount = 0.0
        for rate, amount in zip(np.array(order.tolist())[:, 0], np.array(order.tolist())[:, 1]):
            sumAmount += float(amount)
            sumBtcValue += float(rate) * float(amount)
            if float(btcValue) < sumBtcValue:
                break
        rate = self.floatToEighthDigit(rate)
        coinAmount = self.floatToEighthDigit(sumAmount - (float(sumBtcValue) - float(btcValue)) / float(rate))
        return rate, coinAmount

    def marketMarginBuy(self, coin, btcValue):
        """Buy coin with market price as much as possible."""
        self.cancelOnMarginOrder(coin)
        btcValue = self.floatToEighthDigit(btcValue)
        tradableBalance = self.returnTradableBalance()
        if float(btcValue) > float(tradableBalance):
            btcValue = tradableBalance
        rate, coinAmount = self.returnRateAndAmount("asks", coin, btcValue)
        if float(rate) * float(coinAmount) < 0.0001:
            return
        ret = self.marginBuy(self.basicCoin + "_" + coin, rate, coinAmount, lendingRate=0.02)
        while ret["success"] == 0:
            rate, coinAmount = self.returnRateAndAmount("asks", coin, btcValue)
            ret = self.marginBuy(self.basicCoin + "_" + coin, rate, coinAmount, lendingRate=0.02)
            time.sleep(1)
            btcValue = self.floatToEighthDigit(0.999 * float(btcValue))
        return ret

    def marketMarginSell(self, coin, btcValue):
        """Buy coin with market price as much as possible."""
        self.cancelOnMarginOrder(coin)
        btcValue = self.floatToEighthDigit(btcValue)
        tradableBalance = self.returnTradableBalance()
        if float(btcValue) > float(tradableBalance):
            btcValue = tradableBalance
        rate, coinAmount = self.returnRateAndAmount("bids", coin, btcValue)
        if float(rate) * float(coinAmount) < 0.0001:
            return
        ret = self.marginSell(self.basicCoin + "_" + coin, rate, coinAmount, lendingRate=0.02)
        while ret["success"] == 0:
            rate, coinAmount = self.returnRateAndAmount("bids", coin, btcValue)
            ret = self.marginSell(self.basicCoin + "_" + coin, rate, coinAmount, lendingRate=0.02)
            time.sleep(1)
            btcValue = self.floatToEighthDigit(0.999 * float(btcValue))
        return ret

    def distributedBtcValue(self):
        """Return BTC value that is whole you can trade divide tby the number of coin you want to trade."""
        summary = self.returnSummary()
        return self.floatToEighthDigit(float(summary.loc["netValue"]) * self.leverage / len(self.coins))

    def fitBalance(self):
        """Re-take your positions based on the trading sign."""
        position = self.getOpeningMarginPosition()
        for coinIndex in range(len(self.coins)):
            if type(position)!=pd.core.frame.DataFrame: # Hold nothing today?
                if self.tradeSigns[coinIndex] != "hold":  # Trade sign is not "hold" ?
                    if self.tradeSigns[coinIndex] == "long":  # Trade sign is "long" ?
                        self.marketMarginBuy(self.coins[coinIndex], self.distributedBtcValue())
                    else:
                        self.marketMarginSell(self.coins[coinIndex], self.distributedBtcValue())
            else:
                if self.basicCoin + "_" + self.coins[coinIndex] in position.index:  # Opening the position?
                    if position.loc[self.basicCoin + "_" + self.coins[coinIndex]]["type"] != self.tradeSigns[coinIndex]: # Position type is not the same with trade sign?
                        self.closeMarginPosition(self.basicCoin + "_" + self.coins[coinIndex])
                        if self.tradeSigns[coinIndex] != "hold":  # Trade sign is not "hold" ?
                            if self.tradeSigns[coinIndex] == "long":  # Trade sign is "long" ?
                                self.marketMarginBuy(self.coins[coinIndex], self.distributedBtcValue())
                            else:
                                self.marketMarginSell(self.coins[coinIndex], self.distributedBtcValue())
                else:
                    if self.tradeSigns[coinIndex] != "hold":  # Trade sign is not "hold" ?
                        if self.tradeSigns[coinIndex] == "long":  # Trade sign is "long" ?
                            self.marketMarginBuy(self.coins[coinIndex], self.distributedBtcValue())
                        else:
                            self.marketMarginSell(self.coins[coinIndex], self.distributedBtcValue())

    def closeAllOpeningMarginPosition(self):
        """Close all your positions."""
        position = self.getOpeningMarginPosition()
        if type(position) == pd.core.frame.DataFrame:
            for coinIndex in range(len(position.index)):
                self.closeMarginPosition(position.index[coinIndex])

    def returnEstimatedValueOfHoldings(self):
        """Return EstimatedValueOfHoldings."""
        summary = self.returnSummary()
        estimatedValueOfHoldingsAsBTC = float(summary.loc["netValue"])
        lastValueUSDT_BTC = pd.DataFrame.from_dict(self.returnTicker()).T.loc["USDT_BTC"]["last"]
        estimatedValueOfHoldingsAsUSD = float(lastValueUSDT_BTC) * estimatedValueOfHoldingsAsBTC
        return estimatedValueOfHoldingsAsBTC, estimatedValueOfHoldingsAsUSD

    def getSummary(self):
        """Get your balance and return it as string."""
        myBTC, myUSD = self.returnEstimatedValueOfHoldings()
        summary = self.returnSummary()
        positions = self.getOpeningMarginPosition()
        onOrders = pd.DataFrame.from_dict(self.returnOpenOrders(pair="all"))
        if len(onOrders) == 0:
            onMarginOrders ="Nothing"
        else:
            onMarginOrders = onOrders.loc[np.where(onOrders["margin"] == 1)]
            if len(onMarginOrders) == 0:
                onMarginOrders = "Nothing"

        summaryStr = ""
        summaryStr += "-----------------------------------------\n"
        summaryStr += "Poloniex Margin Account Balance.\n"
        summaryStr += "-----------------------------------------\n"
        summaryStr += "Today: " + self.todayStr + "\n"
        summaryStr += "Coins: " + str(self.coins) + "\n"
        summaryStr += "TradeSigns: " + np.str(self.tradeSigns) + "\n"
        summaryStr += "\n"
        summaryStr += "Your total fund in margin account:\n"
        summaryStr += str(myBTC) + " BTC\n"
        summaryStr += str(myUSD) + " USD\n"
        summaryStr += "\n"
        summaryStr += "Summary:\n"
        summaryStr += str(summary)
        summaryStr += "\n"
        summaryStr += "\n"
        summaryStr += "Breakdown:\n"
        summaryStr += str(positions)
        summaryStr += "\n"
        summaryStr += "\n"
        summaryStr += "On order:\n"
        summaryStr += str(onMarginOrders)
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


    def savePoloniexMarginAccountBalanceToCsv(self):
        """Save EstimatedValueOfHoldings to csv file."""
        fileName = self.workingDirPath + "/PoloniexMarginAccountBalance.csv"
        date = str(datetime.datetime.today())[0:19]
        myBTC, myUSD = self.returnEstimatedValueOfHoldings()
        if os.path.exists(fileName):
            f = open(fileName, "a")
            writer = csv.writer(f, lineterminator="\n")
        else:
            f = open(fileName, "w")
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Date", "BTC", "USD"])  # Write header
        writer.writerow([date, str(myBTC), str(myUSD)])
        f.close()
