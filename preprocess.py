# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:06:28 2022

@author: Kristina
"""

#preprocessing
"""to-do:
    0)read file,
    1)em-dash to space, fuse hyphens DONE
    2)replace microsoft quotation marks and commas
    3)change possessive into "x of y" DONE
    4)"it's" into "it is" DONE
    5) "could've" into could have DONE
"""
import csv
import glob
import string

POSSESSIVE_ADJECTIVES = ["his", "her", "our", "their", "my", "your", "its"]
files = []
for file in glob.glob("*.txt"):
    print(file)
    files.append(file)
    
def load_data(filename):
    file = open(filename, "r", encoding='utf-8-sig')
    newname = "01prepro-" + filename
    newfile = open(newname, "w", encoding ='utf-8-sig')
    filelines = file.readlines()
    for row in filelines:
        if row:
            row = row.lower()
            row = replace_punct(row)
            row = replace_genitive(row) #possessive "x's" into "of x"
            newfile.writelines(row)




def replace_punct(sent):
    sent = sent.replace("’", "'") #apostrophe conversion
    sent = sent.replace("‘", "'") #apostrophe conversion
    sent = sent.replace("“", '"') #remove left and right quotation marks
    sent = sent.replace("”", '"') #remove left and right quotation marks
    sent = sent.replace("—", " ") #em-dash to space
    sent = sent.replace("-","") #remove hyphens
    sent = sent.replace("n't"," not") #n't
    sent = sent.replace("it's", "it is") # "it's" into "it is"
    sent = sent.replace("that's", "that is") # "that's" into "that is"
    sent = sent.replace("what's", "what is") # "what's" into "what is"
    sent = sent.replace("there's", "there is") # "what's" into "what is"
    sent = sent.replace("where's", "where is") # "what's" into "what is"
    sent = sent.replace("could've", "could have") #could've into could have
    sent = sent.replace("should've", "should have") #should've into should have
    sent = sent.replace("would've", "would have") #would've into would have
    return sent
    
def replace_genitive(sent):
    words = sent.split(" ")
    i = 0
    newwords = []
    while i < len(words):
        word = words[i]
        if word.find("'s") >= 0 or word.find("s'") >= 0:
            newwords.append("of")
            if word.find("s'") >= 0:
                newwords.append(word.replace("s'",""))
            else:
                newwords.append(word.replace("'s",""))
        else: 
            newwords.append(word)
        i=i+1
    return " ".join(newwords)

def replace_genitive_old(sent):
    words = sent.split(" ")
    i = 0
    newwords = []
    while i < len(words):
        word = words[i]
        plural = False
        auxword = None
        if word.find("'s") >= 0 or word.find("s'") >= 0:
            if word.find("s'") >= 0:
                plural = True
            if i>0 and words[i-1].lower() in POSSESSIVE_ADJECTIVES:
                auxword = newwords.pop()
            recurs = replace_genitive_word(words,i)
            j = i
            i = i + recurs
            punct = ""
            #no object
            if j+recurs >= len(words):
                newwords.append("of")
                if recurs > 1:
                    punct = chain_possessive(newwords, words, word, recurs-1, j)
                if auxword:
                    newwords.append(auxword)
                if not plural:
                    newwords.append(word[0:-2])
                else:
                    newwords.append(word[0:-1])
            else:
                punct = chain_possessive(newwords, words,word,recurs,j)
                if plural:
                    if auxword:
                        newwords.append(auxword)
                    newwords.append(word[0:-1])
                else:
                    if auxword:
                        newwords.append(auxword)
                    newwords.append(word[0:-2])
            newwords[-1] = newwords[-1]+punct
        else: 
            newwords.append(word)
        i = i+1
    return " ".join(newwords)
            

def replace_genitive_word(words, idx, level=1):
    if idx+1 != len(words):
        if words[idx+1].find("'s") >= 0 or words[idx+1].find("s'") >= 0:
            level = replace_genitive_word(words, idx+1, level+1)
        else:
            return level
    return level

def chain_possessive(newwords, words, word, recurs, j):
                punct = ""
                while recurs > 0:
                    newword = words[j+recurs]
                    if newword.find("'s") >= 0:
                        newwords.append(newword[0:-2])
                    elif newword.find("s'") >= 0:
                        newwords.append(newword[0:-1])
                    else:
                        if newword[-1] in string.punctuation:
                            punct = newword[-1]
                            newword = newword[0:-1]
                        newwords.append(newword)
                    newwords.append("of")
                    recurs = recurs - 1
                return punct
                    


if __name__ == '__main__':
    print(replace_genitive("my daughter's friends' dog, grandmamma's.!"))
    print(replace_punct("his wife against a mistake in the eyes of the world—he had"))
    """ for file in files:
        load_data(file)"""
          