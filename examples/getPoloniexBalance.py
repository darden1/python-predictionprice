# -*- coding: utf-8 -*-
import os
import sys
import logging
import json
import hmac
import hashlib
from time import sleep, time, gmtime, strftime, strptime, localtime, mktime
from calendar import timegm
import numpy as np


# pip
import requests
if sys.version_info[0] is 3:
    from urllib.parse import urlencode
else:
    from urllib import urlencode


def main():
    myAPIKey = "************************"
    mySecret = "************************************************"
    
    polo = Poloniex(APIKey=myAPIKey, Secret=mySecret, timeout=10, coach=True)
    balanceAll = polo.myCompleteBalances(account="exchange")

    myBTC = 0
    for i in range(len(balanceAll.items())):
        myBTC += float(balanceAll.items()[i][1]["btcValue"])

    tmp = polo.marketTicker()
    lastValueUSDT_BTC = 0
    for i, currentPair in enumerate(tmp.keys()):
        if currentPair == "USDT_BTC":
            lastValueUSDT_BTC = float(tmp.values()[i]["last"])
            break
    myUSD = lastValueUSDT_BTC * myBTC
    
    print("-"*35)
    print("Your total fund in exchange account:")
    print(str(myBTC) + " BTC")
    print(str(myUSD) + " USD")
    print("\nBreakdown:")
    print("     available  btcValue  onOrders")
    for i in range(len(balanceAll.items())):
        if balanceAll.items()[i][1]["btcValue"] != "0.00000000":
            tmpStr = ""
            tmpStr += balanceAll.keys()[i] + " "
            tmpStr += balanceAll.values()[i]["available"] + " "
            tmpStr += balanceAll.values()[i]["btcValue"] + " "
            tmpStr += balanceAll.values()[i]["onOrders"]
            print(tmpStr)

# Possible Commands
PUBLIC_COMMANDS = [
    'returnTicker',
    'return24hVolume',
    'returnOrderBook',
    'returnTradeHistory',
    'returnChartData',
    'returnCurrencies',
    'returnLoanOrders']

PRIVATE_COMMANDS = [
    'returnBalances',
    'returnCompleteBalances',
    'returnDepositAddresses',
    'generateNewAddress',
    'returnDepositsWithdrawals',
    'returnOpenOrders',
    'returnTradeHistory',
    'returnAvailableAccountBalances',
    'returnTradableBalances',
    'returnOpenLoanOffers',
    'returnOrderTrades',
    'returnActiveLoans',
    'createLoanOffer',
    'cancelLoanOffer',
    'toggleAutoRenew',
    'buy',
    'sell',
    'cancelOrder',
    'moveOrder',
    'withdraw',
    'returnFeeInfo',
    'transferBalance',
    'returnMarginAccountSummary',
    'marginBuy',
    'marginSell',
    'getMarginPosition',
    'closeMarginPosition']

class Poloniex(object):
    """The Poloniex Object!"""
    def __init__(
            self, APIKey=False, Secret=False,
            timeout=3, coach=False, loglevel=logging.WARNING):
        """
        APIKey = str api key supplied by Poloniex
        Secret = str secret hash supplied by Poloniex
        timeout = int time in sec to wait for an api response
            (otherwise 'requests.exceptions.Timeout' is raised)
        coach = bool to indicate if the api coach should be used
        loglevel = logging level object to set the module at
            (changes the requests module as well)

        self.apiCoach = object that regulates spacing between api calls

        # Time Placeholders # (MONTH == 30*DAYS)

        self.MINUTE, self.HOUR, self.DAY, self.WEEK, self.MONTH, self.YEAR
        """
        # Set wrapper logging level
        logging.basicConfig(
                format='[%(asctime)s] %(message)s',
                datefmt="%H:%M:%S",
                level=loglevel)
        # Suppress the requests	module logging output
        logging.getLogger("requests").setLevel(loglevel)
        logging.getLogger("urllib3").setLevel(loglevel)
        # Call coach, set nonce
        self.apiCoach, self.nonce = Coach(), int(time()*1000)
        # Grab keys, set timeout, ditch coach?
        self.APIKey, self.Secret, self.timeout, self._coaching = \
            APIKey, Secret, timeout, coach
        # Set time labels
        self.MINUTE, self.HOUR, self.DAY, self.WEEK, self.MONTH, self.YEAR = \
            60, 60*60, 60*60*24, 60*60*24*7, 60*60*24*30, 60*60*24*365

    # -----------------Meat and Potatos---------------------------------------
    def api(self, command, args={}):
        """
        Main Api Function
        - encodes and sends <command> with optional [args] to Poloniex api
        - raises 'ValueError' if an api key or secret is missing
            (and the command is 'private'), or if the <command> is not valid
        - returns decoded json api message
        """
        global PUBLIC_COMMANDS, PRIVATE_COMMANDS

        # check in with the coach
        if self._coaching:
            self.apiCoach.wait()

        # pass the command
        args['command'] = command

        # private?
        if command in PRIVATE_COMMANDS:
            # check for keys
            if not self.APIKey or not self.Secret:
                raise ValueError("APIKey and Secret needed!")
            # set nonce
            args['nonce'] = self.nonce

            try:
                # encode arguments for url
                postData = urlencode(args)
                # sign postData with our Secret
                sign = hmac.new(
                        self.Secret.encode('utf-8'),
                        postData.encode('utf-8'),
                        hashlib.sha512)
                # post request
                ret = requests.post(
                        'https://poloniex.com/tradingApi',
                        data=args,
                        headers={
                            'Sign': sign.hexdigest(),
                            'Key': self.APIKey
                            },
                        timeout=self.timeout)
                # return decoded json
                return json.loads(ret.text)

            except Exception as e:
                raise e

            finally:
                # increment nonce(no matter what)
                self.nonce += 1

        # public?
        elif command in PUBLIC_COMMANDS:
            try:
                ret = requests.post(
                        'https://poloniex.com/public?' + urlencode(args),
                        timeout=self.timeout)
                return json.loads(ret.text)
            except Exception as e:
                raise e
        else:
            raise ValueError("Invalid Command!")

    # --PUBLIC COMMANDS-------------------------------------------------------
    def marketTicker(self):
        """ Returns the ticker for all markets """
        return self.api('returnTicker')

    def marketVolume(self):
        """ Returns the volume data for all markets """
        return self.api('return24hVolume')

    def marketStatus(self):
        """ Returns additional market info for all markets """
        return self.api('returnCurrencies')

    def marketLoans(self, coin):
        """ Returns loan order book for <coin> """
        return self.api('returnLoanOrders', {'currency': str(coin)})

    def marketOrders(self, pair='all', depth=20):
        """
        Returns orderbook for [pair='all']
        at a depth of [depth=20] orders
        """
        return self.api('returnOrderBook', {
                    'currencyPair': str(pair),
                    'depth': str(depth)
                    })

    def marketChart(self, pair, period=False, start=False, end=time()):
        """
        Returns chart data for <pair> with a candle period of
        [period=self.DAY] starting from [start=time()-self.YEAR]
        and ending at [end=time()]
        """
        if not period:
            period = self.DAY
        if not start:
            start = time()-(self.MONTH*2)
        return self.api('returnChartData', {
                    'currencyPair': str(pair),
                    'period': str(period),
                    'start': str(start),
                    'end': str(end)
                    })

    def marketTradeHist(self, pair, start=False, end=time()):
        """
        Returns public trade history for <pair>
        starting at <start> and ending at [end=time()]
        """
        if self._coaching:
            self.apiCoach.wait()
        if not start:
            start = time()-self.HOUR
        try:
            ret = requests.post(
                    'https://poloniex.com/public?'+urlencode({
                        'command': 'returnTradeHistory',
                        'currencyPair': str(pair),
                        'start': str(start),
                        'end': str(end)
                        }),
                    timeout=self.timeout)
            return json.loads(ret.text)
        except Exception as e:
            raise e

    # --PRIVATE COMMANDS------------------------------------------------------
    def myTradeHist(self, pair):
        """ Returns private trade history for <pair> """
        return self.api('returnTradeHistory', {'currencyPair': str(pair)})

    def myBalances(self):
        """ Returns coin balances """
        return self.api('returnBalances')

    def myAvailBalances(self):
        """ Returns available account balances """
        return self.api('returnAvailableAccountBalances')

    def myMarginAccountSummary(self):
        """ Returns margin account summary """
        return self.api('returnMarginAccountSummary')

    def myMarginPosition(self, pair='all'):
        """ Returns margin position for [pair='all'] """
        return self.api('getMarginPosition', {'currencyPair': str(pair)})

    def myCompleteBalances(self, account='all'):
        """ Returns complete balances """
        return self.api('returnCompleteBalances', {'account': str(account)})

    def myAddresses(self):
        """ Returns deposit addresses """
        return self.api('returnDepositAddresses')

    def myOrders(self, pair='all'):
        """ Returns your open orders for [pair='all'] """
        return self.api('returnOpenOrders', {'currencyPair': str(pair)})

    def myDepositsWithdraws(self):
        """ Returns deposit/withdraw history """
        return self.api('returnDepositsWithdrawals')

    def myTradeableBalances(self):
        """ Returns tradable balances """
        return self.api('returnTradableBalances')

    def myActiveLoans(self):
        """ Returns active loans """
        return self.api('returnActiveLoans')

    def myOpenLoanOrders(self):
        """ Returns open loan offers """
        return self.api('returnOpenLoanOffers')

    # --Trading functions-- #
    def orderTrades(self, orderId):
        """ Returns any trades made from <orderId> """
        return self.api('returnOrderTrades', {'orderNumber': str(orderId)})

    def createLoanOrder(self, coin, amount, rate, autoRenew=0, duration=2):
        """ Creates a loan offer for <coin> for <amount> at <rate> """
        return self.api('createLoanOffer', {
                    'currency': str(coin),
                    'amount': str(amount),
                    'duration': str(duration),
                    'autoRenew': str(autoRenew),
                    'lendingRate': str(rate)
                    })

    def cancelLoanOrder(self, orderId):
        """ Cancels the loan offer with <orderId> """
        return self.api('cancelLoanOffer', {'orderNumber': str(orderId)})

    def toggleAutoRenew(self, orderId):
        """ Toggles the 'autorenew' feature on loan <orderId> """
        return self.api('toggleAutoRenew', {'orderNumber': str(orderId)})

    def closeMarginPosition(self, pair):
        """ Closes the margin position on <pair> """
        return self.api('closeMarginPosition', {'currencyPair': str(pair)})

    def marginBuy(self, pair, rate, amount, lendingRate=2):
        """ Creates <pair> margin buy order at <rate> for <amount> """
        return self.api('marginBuy', {
                    'currencyPair': str(pair),
                    'rate': str(rate),
                    'amount': str(amount),
                    'lendingRate': str(lendingRate)
                    })

    def marginSell(self, pair, rate, amount, lendingRate=2):
        """ Creates <pair> margin sell order at <rate> for <amount> """
        return self.api('marginSell', {
                    'currencyPair': str(pair),
                    'rate': str(rate),
                    'amount': str(amount),
                    'lendingRate': str(lendingRate)
                    })

    def buy(self, pair, rate, amount):
        """ Creates buy order for <pair> at <rate> for <amount> """
        return self.api('buy', {
                    'currencyPair': str(pair),
                    'rate': str(rate),
                    'amount': str(amount)
                    })

    def sell(self, pair, rate, amount):
        """ Creates sell order for <pair> at <rate> for <amount> """
        return self.api('sell', {
                    'currencyPair': str(pair),
                    'rate': str(rate),
                    'amount': str(amount)
                    })

    def cancelOrder(self, orderId):
        """ Cancels order <orderId> """
        return self.api('cancelOrder', {'orderNumber': str(orderId)})

    def moveOrder(self, orderId, rate, amount):
        """ Moves an order by <orderId> to <rate> for <amount> """
        return self.api('moveOrder', {
                    'orderNumber': str(orderId),
                    'rate': str(rate),
                    'amount': str(amount)
                    })

    def withdraw(self, coin, amount, address):
        """ Withdraws <coin> <amount> to <address> """
        return self.api('withdraw', {
                    'currency': str(coin),
                    'amount': str(amount),
                    'address': str(address)
                    })

    def returnFeeInfo(self):
        """ Returns current trading fees and trailing 30-day volume in BTC """
        return self.api('returnFeeInfo')

    def transferBalance(self, coin, amount, fromac, toac):
        """
        Transfers coins between accounts (exchange, margin, lending)
        - moves <coin> <amount> from <fromac> to <toac>
        """
        return self.api('transferBalance', {
                    'currency': str(coin),
                    'amount': str(amount),
                    'fromAccount': str(fromac),
                    'toAccount': str(toac)
                    })

# Convertions
def epoch2UTCstr(timestamp=time(), fmat="%Y-%m-%d %H:%M:%S"):
    """
    - takes epoch timestamp
    - returns UTC formated string
    """
    return strftime(fmat, gmtime(timestamp))

def UTCstr2epoch(datestr=epoch2UTCstr(), fmat="%Y-%m-%d %H:%M:%S"):
    """
    - takes UTC date string
    - returns epoch
    """
    return timegm(strptime(datestr, fmat))

def epoch2localstr(timestamp=time(), fmat="%Y-%m-%d %H:%M:%S"):
    """
    - takes epoch timestamp
    - returns localtimezone formated string
    """
    return strftime(fmat, localtime(timestamp))

def localstr2epoch(datestr=epoch2UTCstr(), fmat="%Y-%m-%d %H:%M:%S"):
    """
    - takes localtimezone date string,
    - returns epoch
    """
    return mktime(strptime(datestr, fmat))

def float2roundPercent(floatN, decimalP=2):
    """
    - takes float
    - returns percent(*100) rounded to the Nth decimal place as a string
    """
    return str(round(float(floatN)*100, decimalP))+"%"

# Coach
class Coach(object):
    """
    Coaches the api wrapper, makes sure it doesn't get all hyped up on Mt.Dew
    Poloniex default call limit is 6 calls per 1 sec.
    """
    def __init__(self, timeFrame=1.0, callLimit=6):
        """
        timeFrame = float time in secs [default = 1.0]
        callLimit = int max amount of calls per 'timeFrame' [default = 6]
        """
        self._timeFrame, self._callLimit = timeFrame, callLimit
        self._timeBook = []

    def wait(self):
        """ Makes sure our api calls don't go past the api call limit """
        # what time is it?
        now = time()
        # if it's our turn
        if len(self._timeBook) is 0 or \
                (now - self._timeBook[-1]) >= self._timeFrame:
            # add 'now' to the front of 'timeBook', pushing other times back
            self._timeBook.insert(0, now)
            logging.info(
                "Now: %d  Oldest Call: %d  Diff: %f sec" %
                (now, self._timeBook[-1], now - self._timeBook[-1])
                )
            # 'timeBook' list is longer than 'callLimit'?
            if len(self._timeBook) > self._callLimit:
                # remove the oldest time
                self._timeBook.pop()
        else:
            logging.info(
                "Now: %d  Oldest Call: %d  Diff: %f sec" %
                (now, self._timeBook[-1], now - self._timeBook[-1])
                )
            logging.info(
                "Waiting %s sec..." %
                str(self._timeFrame-(now - self._timeBook[-1]))
                )
            # wait your turn (maxTime - (now - oldest)) = time left to wait
            sleep(self._timeFrame-(now - self._timeBook[-1]))
            # add 'now' to the front of 'timeBook', pushing other times back
            self._timeBook.insert(0, time())
            # 'timeBook' list is longer than 'callLimit'?
            if len(self._timeBook) > self._callLimit:
                # remove the oldest time
                self._timeBook.pop()


if __name__ == "__main__":
    main()
