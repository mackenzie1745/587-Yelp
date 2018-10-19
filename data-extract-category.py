#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:23:20 2018

@author: fraifeld-mba
"""

import pandas as pd
import os
import json
import time
import os, psutil

    
    

def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def extract_bus_AZ_PA(head):
    if head:
        return pd.read_json("business_head.json", lines = True)
    else: 
        count = 0
        with open("business.json", "r") as f:
            with open("business-AZ-PA.json", "w+") as w:
                for line in f:
                    count = count+1
                    j = json.loads(line)
                    print(str(count) + j["name"])
                    try:
                        if j["state"] in ["AZ","PA"]:
                            w.write(line)
                    except:
                        print(j)
                        print("stop? (y/n)")
                        response = input()
                        if response == "y":
                            break
        
                        
                    
        
def extract_rev_AZ_PA(business_ids, filename, head):
    if head:
        return pd.read_json("review_head.json", lines = True)
    else: 
        count = 0
        with open("review.json", "r") as f:
            with open("filename", "w+") as w:
                for line in f:
                    count = count+1
                    print(count)
                    j = json.loads(line)
                    try:
                        if j["business_id"] in business_ids:
                            w.write(line)
                    except:
                        print(j)
                        print("stop? (y/n)")
                        response = input()
                        if response == "y":
                            break

# Same function as above but with different filenames.


def extract_bus_restaurants():
    bus = get_businesses_df()
    restaurants = bus[bus["categories"].notnull()]
    restaurants = bus[bus["categories"].str.contains("Restaurants", na=False)]
    restaurants.to_json("business-Restaurants-AZ-PA.json",orient="records", lines=True)


def get_businesses_df():
        return pd.read_json("business-Restaurants-AZ-PA.json",lines=True)

def get_review_df():
        return pd.read_json("review-Restaurants-AZ-PA.json",lines=True)        

def get_categories():
    with open("categories.txt", "r") as categories:
        for category in categories:
            print(category)


def merge(n):
    df = get_businesses_df()
    with open("review-Restaurants-AZ-PA.json","r") as r:
        with open("merge2.json","w+") as w:
            count = 0
            for line in r:
                count = count + 1
                j = json.loads(line)
                business_id = j["business_id"]
                df[df["business_id"]==business_id]
                row = {}
                for i in j:
                    row[i] = j[i]
                for i in df[df["business_id"]==business_id].columns:
                    row[i] = str(df[df["business_id"]==business_id].reset_index().loc[0,i])
                print(row["name"])
                w.write(str(json.dumps(row))+"\n")
                if count > n:
                    break
        
            



if __name__ == "__main__":
    #extract_bus_AZ_PA(head = False)
    #df = extract_bus_restaurants()
    #business_ids = list(df["business_id"])
    #extract_rev_AZ_PA(business_ids, "review-Restaurants-AZ-PA.json",head = False)
    merge(150000)
