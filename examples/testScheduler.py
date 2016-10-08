# -*- coding: utf-8 -*-
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler

def main():
    sc = BlockingScheduler(timezone='UTC')
    sc.add_job(getPath, 'cron', hour=0, minute=5)
    sc.start()

def getPath():
    print os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main()


