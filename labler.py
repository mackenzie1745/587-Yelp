#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:25:25 2018

@author: fraifeld-mba
"""

import json

def parse_json(line):
    j = json.loads(line)
    return j

def ask(j):
    print(j["text"] + "\n" + "s, f, n, or b:")


def solicit_answer():
    answer = input().lower()
    while (answer not in ["s","f","n","b"]):
        print("Please try again: s, f, n, or b")
        answer = input().lower()
    print("----------\n")
    return answer
        
def append_answer(j, answer):
    j["label"] = answer
    return j

def label(f, w, i, k):
    count = 1
    for line in f:
        if count < i:
            pass
        elif count < k:
            j = parse_json(line)
            ask(j)
            answer = solicit_answer()
            w.write(json.dumps(append_answer(j, answer))+"\n")
        else:
            print("At index " + str(k) + "... exiting.")
            break
        count = count + 1

            

if __name__ == "__main__":
    print("Please specify the index of the first review you want to label")
    i = int(input())
    print("Please specify the index of the last review you want to label")
    j = int(input())
    print("----------\nLabeling Reviews \nDictionary: \ns = service \nf = food \nb = both \nn = none" )
    print("----------\nPlease ignore what I said earlier. If there is any combination of food and service, label it as both. We will deal with the both's seperately \n----------")
    with open("merge2.json","r") as f:
        with open("labeled_reviews.json","w+") as w:
            label(f, w, i, j)