# -*- coding: utf-8 -*-
import email
import smtplib
import numpy as np
from predictionprice import CustumPoloniex
from predictionprice import PredictionPrice


def main():
    myGmailAddress = "********@gmail.com"
    myGmailAddressPassword = "************"
    myAPIKey="************************"
    mySecret="************************************************"
    
    basicCoin="BTC"
    workingDirPath="."
    
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
    for coinIndex in range(len(coins)):
        pp = PredictionPrice(currentPair=basicCoin+"_"+coins[coinIndex], workingDirPath=workingDirPath,
                             gmailAddress=myGmailAddress, gmailAddressPassword=myGmailAddressPassword,
                             backTestOptNumFeatureMin=backTestOptParams[coinIndex][0], backTestOptNumFeatureMax=backTestOptParams[coinIndex][1],
                             backTestOptNumTrainSampleMin=backTestOptParams[coinIndex][2], backTestOptNumTrainSampleMax=backTestOptParams[coinIndex][3])

        pp.fit(pp.appreciationRate_,pp.quantizer(pp.appreciationRate_))
        pp.sendMail(pp.getComment())
        ppList.append(pp)
        tommorrwPricePrediction.append(pp.tommorrowPriceFlag_)
        
    #--- Fit balance
    if len(tommorrwPricePrediction)==len(coins): #No error
        polo = CustumPoloniex(APIKey=myAPIKey, Secret=mySecret, timeout=10, coach=True, basicCoin=basicCoin)
        fitSell(polo,coins,tommorrwPricePrediction)
        fitBuy(polo,coins,tommorrwPricePrediction)
        balance=polo.myAvailableCompleteBalances()
        myBTC,myUSD=polo.myEstimatedValueOfHoldings()

        comStr=""
        comStr += "Your Fund is " + str(myBTC) + " BTC\n"
        comStr += "Your Fund is " + str(myUSD) + " USD\n"
        comStr += str(balance)

        sendMailBalance(comStr,myGmailAddress,myGmailAddressPassword )

    #--- back test optimization
    for coinIndex in range(len(coins)):
        pp = ppList[coinIndex]
        pp.backTestOptimization(pp.appreciationRate_, pp.quantizer(pp.appreciationRate_))


def sendMailBalance(body,gmailAddress, gmailAddressPassword):
    if gmailAddress=="" or gmailAddressPassword=="":
        return "Set your gmail address and password."
    # ---Create message
    msg = email.MIMEMultipart.MIMEMultipart()
    msg["From"] = gmailAddress
    msg["To"] = gmailAddress
    msg["Date"] = email.Utils.formatdate()
    msg["Subject"] = "Poloniex Balance"
    msg.attach(email.MIMEText.MIMEText(body))
    # ---SendMail
    smtpobj = smtplib.SMTP("smtp.gmail.com", 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.login(gmailAddress, gmailAddressPassword)
    smtpobj.sendmail(gmailAddress, gmailAddress, msg.as_string())
    smtpobj.close()

def fitSell(polo,coins,tommorrwPricePrediction):
    balance=polo.myAvailableCompleteBalances()    
    
    for coinIndex in range(len(coins)):
        if not tommorrwPricePrediction[coinIndex]: #Prediction is Sell?
            if not (len(np.where(balance.index==coins[coinIndex])[0])==0): #Holding the coin?
                polo.marketSellAll(coins[coinIndex])

def fitBuy(polo,coins,tommorrwPricePrediction):
    balance=polo.myAvailableCompleteBalances()    
    if np.sum(tommorrwPricePrediction)==0: # Signs are all sell?
        return
    else:
        myBTC,myUSD=polo.myEstimatedValueOfHoldings()
        distributionBTCValue = myBTC*1.0/np.sum(tommorrwPricePrediction)
        
        for coinIndex in range(len(coins)):
            if tommorrwPricePrediction[coinIndex]: #Prediction is Buy?
                if not (len(np.where(balance.index==coins[coinIndex])[0])==0): #Holding the coin?
                    extraBTCValue=float(balance.loc[coins[coinIndex]]["btcValue"])-float(distributionBTCValue)
                    if extraBTCValue>0:
                        polo.marketSell(coins[coinIndex],np.abs(extraBTCValue))
                    else:
                        polo.marketBuy(coins[coinIndex],np.abs(extraBTCValue))
                else:
                    polo.marketBuy(coins[coinIndex],distributionBTCValue)
                    
if __name__ == "__main__":
    main()





