# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 08:36:13 2022

@author: Kristina
"""

import csv
import numpy
import math
import pickle
import glob
import string
#filename = "output-karenina.txt"

filenames = []
for file in glob.glob("*.txt"):
    print(file)
    filenames.append(file)

stemmingdic = {}
countdic = {}


for filename in filenames:
    f = open(filename, "r", encoding='utf-8-sig', newline='')
    file = csv.reader(f, delimiter="\t")
    i = 0
    previous = ""
    for row in file:
        if (len(row)==3):
            word = row[0].lower().replace("_","")
            stem = row[2]
            stem = stem.lower()
            stem = stem.translate(str.maketrans('', '', string.punctuation.replace("<","").replace(">","").replace("|","")))
            """if stem == "n't":
                stem = "not"
                word = "not"""
            if len(word) >= 1 and word[0] == "'":
                stem = previous+word
                word = stem
                countdic[previous] = countdic[previous]-1
                countdic.update({previous:countdic[previous]})
            elif stem == "<unknown>" and not set(word).intersection(set(string.punctuation)):
                stem = word
                print(stem + " 1")
            elif stem == "<unknown>":
                print(word + " 2")
                stem = ""
            elif len(word) == 1 and set(word).intersection(set(string.punctuation)):
                stem = ""
            if "|" in stem:
                idx = stem.find("|")
                stem = stem[0:idx]
            word = word.translate(str.maketrans('', '', string.punctuation))
            stem = stem.translate(str.maketrans('', '', string.punctuation))
            word = word.translate(str.maketrans('', '', "—’“”"))
            if stem:
                stemmingdic.update({word:stem})
                if stem in countdic:
                    count = countdic[stem] + 1
                    countdic.update({stem:count})
                else:
                    countdic.update({stem:1})
                previous = stem
        i = i+1
    f.close()




print(stemmingdic)
print(countdic)

