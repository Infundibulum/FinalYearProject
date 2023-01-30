# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 07:48:12 2022

@author: Kristina
"""
import glob
import spacy
import csv
import numpy
import math
import string
import pickle
import sys
from nltk import tokenize
from numpy import linalg
tiebreaker_first = sys.argv[1]
tiebreaker_second = sys.argv[2]
nlp = spacy.load("en_core_web_sm")

def remove_blank(text):
    text = text[0].translate(str.maketrans('', '', "[\]"))
    if not text:
        print('none')
        return ''
    else:
        return tokenize.sent_tokenize(str(text))

files = []
for file in glob.glob("*.txt"):
    print(file)
    files.append(file)

def ispunct(char):
    return char in string.punctuation
    
def load_data(filename):
    file = open(filename, "r", encoding='utf-8-sig')
    sents = []
    lines = []
    filelines = file.readlines()
    for line in filelines:
        line = line.rstrip()+" "
        lines.append(line)
    text = "".join(lines)
    sents = tokenize.sent_tokenize(text)
    print(sents[0:5])
    return sents

def sentence_lemmas(sentence):
    lemmas = []
    punct = string.punctuation + '“”' + "_"
    sent = sentence.translate(str.maketrans('', '', punct)).lower().split()
    stemdic = pickle.load(open("stemmingdic3.1.p", "rb"))
    for word in sent:
        if word in stemdic:
            lemmas.append(stemdic[word])
        else:
            print(word)
    return lemmas

def wordlist(lemmas):
    wlist = []
    for sent in lemmas:
        for word in sent:
            if word not in wlist:
                wlist.append(word)
    wlist.sort()
    wordDict0 = { i : wlist[i] for i in range(0, len(wlist))}
    wordDict = {v: k for k, v in wordDict0.items()}
    wordDict0.update(wordDict)
    return wordDict0

def co_occurrence_matrix(dataset, n_window):
    """
    dataset - list of preprocessed sentences
    n-window - window size

    Returns nxn numpy array -n is the nr of words in dictionary, each field contains co-occurrence 
    counts

    """
    dic = wordlist(dataset)
    n = (int) (len(dic) / 2)
    com = numpy.zeros(shape=(n,n))
    for i in range(n):
        for s in dataset:
            word = dic[i]
            if word in s:
                s.count(word)
                j = -1
                for x in range(s.count(word)):
                    j = s.index(word, j+1)
                    for y in range(1,n_window+1):
                        if j+y < len(s):
                            new = s[j+y]
                            newn = dic[new]
                            com[i,newn] = com[i,newn] + 1
                        if j-y >= 0:
                            new = s[j-y]
                            newn = dic[new]
                            com[i,newn] = com[i,newn] + 1
    return com
    
def ppmi_matrix(cooc_matrix):
    #return n x n numpy array representing the ppmi matrix
    n = cooc_matrix.shape[0]
    countAll = numpy.sum(numpy.sum(cooc_matrix))
    ppmim = numpy.zeros(shape=(n,n))
    #i
    for i in range(0,n):
        for j in range(i,n):
            pi = numpy.sum(cooc_matrix[i]) / countAll
            pj = numpy.sum(cooc_matrix[j]) / countAll
            pij = cooc_matrix[i,j]/countAll
            if pi*pj != 0 and pij != 0:
                ppmi = max(math.log2(pij/(pi*pj)), 0)
            else:
                ppmi = 0
            ppmim[i,j] = ppmi
            ppmim[j,i] = ppmi
    return ppmim

def ppmi(word, context, ppmi_mtx, dataset):
    dic = wordlist(dataset)
    return ppmi_mtx[dic[word],dic[context]]

def get_word_vectors(ppmi_mtx, dataset):
    #get word vectors from ppmi matrix and the list of preprocessed sentences
    dic = wordlist(dataset)
    n = ppmi_mtx.shape[0]
    wordVecs = {}
    for i in range(n):
        word = dic[i]
        wordVec = ppmi_mtx[i]
        wordVecs.update({word: wordVec})
    return wordVecs

def frequency_matrix(n, m, countdic, sentences):
    freq_mat = numpy.zeros((len(countdic), 2*n*m))
    most_common = get_most_common(m, countdic)
    j = 0
    for foc_word in countdic:
        i = 0
        for con_word in most_common:
            for pos in range(0,2*n,2):
                npos = int (pos/2)+1 #pos counted
                countleft = count_n_pos(foc_word, con_word, npos, sentences, False)
                countright = count_n_pos(foc_word, con_word, npos, sentences)
                #even left
                #odd right
                if countdic[foc_word]:
                    freq_mat[j, i+pos] = countleft /countdic[foc_word]
                    freq_mat[j, i+pos+1] = countright /countdic[foc_word]
            i+=2*n
        j+=1
    return freq_mat

def get_most_common(m, countdic):
    counts = []
    most_common = []
    
    def replace(word, oldword, lowcount, wordcount):
        most_common.remove(oldword)
        most_common.append(word)
        counts.remove(lowcount)
        counts.append(wordcount)
    for word in countdic:
        if counts:
            lowcount = min(counts)
        else:
            lowcount = sys.maxsize
        wordcount = countdic[word]
        if len(most_common) < m:
            most_common.append(word)
            counts.append(wordcount)
        elif wordcount == lowcount:
             for common in most_common:
                if countdic[common] == lowcount:
                    if tiebreaker_first == "short" and len(word)<len(common):
                        replace(word,common,lowcount,wordcount)
                        break
                    elif tiebreaker_first =="long" and len(word)>len(common):
                        replace(word,common,lowcount,wordcount)
                        break
                    elif tiebreaker_second == "first" and len(word) == len(common):
                        break
                    elif len(word)==len(common):
                        replace(word,common,lowcount,wordcount)
                        break
        elif wordcount > lowcount:
            for common in most_common:
                if countdic[common] == lowcount:
                    most_common.remove(common)
                    most_common.append(word)
                    counts.remove(lowcount)
                    counts.append(wordcount)
                    break
    return most_common

def count_n_pos(word, conword, n, sentences, right = True):
    count = 0
    multiplier = 1
    if not right:
        multiplier = -1
    for i in range(len(sentences)):
        if word in sentences[i]:
            word_i = sentences[i].index(word)
            conword_i = word_i + multiplier*n
            if right and len(sentences[i]) > conword_i:
                if sentences[i][conword_i] == conword:
                    count += 1
            elif not right and conword_i >= 0:
                if sentences[i][conword_i] == conword:
                    count += 1
    return count

def cosine_sim(vec1, vec2):
        normproduct = linalg.norm(vec1)*linalg.norm(vec2)
        dotproduct = numpy.dot(vec1,vec2)
        return dotproduct/normproduct

def get_n_similar(vecdic, n, word):
    if word in vecdic:
        wordvec = vecdic[word]
        simdic = {}
        simwords = []
        similarity = []  
        lowsim = sys.maxsize
        for newword in vecdic:
            simvec = cosine_sim(vecdic[newword], wordvec)
            if similarity:
                lowsim = min(similarity)
            if len(simwords) < n:
                    simdic.update({newword:simvec})
                    simwords.append(newword)
                    similarity.append(simvec)
            elif simvec > lowsim:
                for sim in simwords:
                    if simdic[sim] == lowsim:
                        simdic.pop(sim)
                        simdic.update({newword:simvec})
                        simwords.remove(sim)
                        simwords.append(newword)
                        similarity.remove(lowsim)
                        similarity.append(simvec)
                        break
        return simwords, similarity
    else:
        return -1
    
def get_n_singletons(countdic, n):
    singles = []
    for word in countdic:
        if countdic[word] == 1:
            singles.append(word)
        if len(singles) == n:
            break
    return singles

def sort_similarity(word, simwords, similarity):
    if word in simwords:
        j = simwords.index(word)
        del(simwords[j])
        del(similarity[j])
    dic = {simwords[i]: similarity[i] for i in range(len(simwords))}
    quicksort(similarity, 0, len(similarity)-1)
    newsim = [None] * len(simwords)
    for i in range(len(simwords)):
        for j in range(len(similarity)):
            if dic[simwords[i]] == similarity[j]:
                newsim[j] = simwords[i]
    return newsim, similarity
                

def quicksort(A, low, high):
    if low >= 0 and high >= 0 and low < high:
        p = partition(A, low, high)
        quicksort(A, low, p)
        quicksort(A, p + 1, high) 

def partition(A, low, high):
  pivot = A[math.floor((high + low) / 2)]
  i = low - 1
  j = high + 1
  while(True):
    while(True):
        i = i+1
        if math.isnan(pivot) and math.isnan(A[i]):
            break
        if A[i] >= pivot:
            break
    while(True):
        j = j-1
        if math.isnan(pivot) and math.isnan(A[i]):
            break
        if A[j] <= pivot:
            break
    if i >= j:
        return j  
    temp = A[i]
    A[i] = A[j]
    A[j] = temp

def write_csv(countdic, word, words, sim, filename):
    f = open(filename, "a")
    delim = ","
    f.write(word + delim + str(countdic[word]))
    for i in range(len(sim)-1,-1,-1):
        f.write(delim+words[i]+delim+str(countdic[words[i]])+delim+str(sim[i]))
    f.write("\n")
    f.close()
    
if __name__ == '__main__':
    """
    n = 3
    sentlist = [] #list of preprocessed sentences
    for file in files:
        filetxt = load_data(file)
        filetxt[:] = [sent for sent in filetxt if sent]
        print(filetxt[0:5])
        for lis in filetxt:
            for sent in lis:
                sentlist.append(sentence_lemmas(sent))
        print("file processed!")
    pickle.dump(sentlist, open( "sentlist.p", "wb" ) )
    cooc_mtx = co_occurrence_matrix(sentlist, n)
    print("cooc matrix done")
    ppmi_mtx = ppmi_matrix(cooc_mtx)
    print("ppmi matrix done!")
    numpy.save('ppmi.npy', ppmi_mtx)
    print("ppmi exported")
    word_vectors = get_word_vectors(ppmi_mtx, sentlist)
    print(word_vectors['happy'])
    """
    """
    sentlist = [] #list of preprocessed sentences
    for file in files:
        filetxt = load_data(file)
        #filetxt[:] = [sent for sent in filetxt if sent]
        #print(filetxt[0:5])
        #print(filetxt)
        for lis in filetxt:
            sentlist.append(sentence_lemmas(lis))
        print("file processed!")
    pickle.dump(sentlist, open( "sentlist3.p", "wb" ) )"""

"""
sentences = [["I", "like", "lovely", "dog", "that", "bark"],["smelly", "dog", "bark", "too"]]
countdic = {"I":1, "like":1, "lovely":1, "dog": 2, "that":1, "bark":2, "smelly":1, "too":1, "cat":1}
print(count_n_pos("dog","smelly",1,sentences, False))
print(get_most_common(3, countdic))
print(frequency_matrix(2, 2, countdic, sentences))
print(count_n_pos("too", "bark", 1, sentences, False))
"""
"""
countdic = pickle.load(open("countdic.p", "rb"))
sentences = pickle.load(open("sentlist2.p", "rb"))
n = 2
m = 30
freq_mat = frequency_matrix(n, m, countdic, sentences)
pickle.dump(freq_mat, open("freqmat30shortfirst.p", "wb"))
"""
"""
"""
"""
freq_mat = pickle.load(open("freqmat30shortfirst3.1.p", "rb"))
countdic = pickle.load(open("countdic3.1.p", "rb"))
vecdic = {}

i=0
for word in countdic:
    vecdic.update({word:freq_mat[i]})
    i +=1

pickle.dump(vecdic, open( "vecdic303.1.p", "wb" ) )
"""
"""
countdic = pickle.load(open("countdic.p", "rb"))
vecdic = pickle.load(open("vecdic30.p", "rb"))

print(get_most_common(200, countdic))
open_class = ["be", "have", "do", "say", "come", "know", "levin", "see", "go"]
closed_class = ["the", "of", "in", "and", "to", "i", "she", "he", "a"]
"""""""
for word in open_class:
    print(word, get_n_similar(vecdic, 4, word))
"""
"""
singletons = get_n_singletons(countdic, 20)
for word in singletons:
    print(word, get_n_similar(vecdic, 4, word))

simwords = ["project", "gutenberg"

]

for word in simwords:
    print(word, " ", countdic[word])
""""""
happy = vecdic["happy"]
print(happy)
print(vecdic["happy"].shape)
sad = vecdic["sad"]
cosim1 = cosine_sim(happy, sad)
print(cosim1)
river = vecdic["laugh"]
cosim2 = cosine_sim(happy,river)
print(cosim2)
"""
"""
#print(load_data("pooh.txt"))
sentlist = [] #list of preprocessed sentences
for file in files:
    filetxt = load_data(file)
    #filetxt[:] = [sent for sent in filetxt if sent]
    #print(filetxt[0:5])
    #print(filetxt)
    for lis in filetxt:
        sentlist.append(sentence_lemmas(lis))
    print("file processed!")
print(sentlist)
pickle.dump(sentlist, open( "sentlist3.1.p", "wb" ) )
"""""""
countdic = pickle.load(open("countdic3.1.p", "rb"))
sentences = pickle.load(open("sentlist3.1.p", "rb"))
n = 2
m = 30
print(countdic["unknown"])
freq_mat = frequency_matrix(n, m, countdic, sentences)
pickle.dump(freq_mat, open("freqmat30shortfirst3.1.p", "wb"))
"""
"""vecdic = pickle.load(open("vecdic302.0.p", "rb"))
countdic = pickle.load(open("countdic3.p", "rb"))"""
"""closed_class = ['the', 'of', 'in', 'and', 'not', 'to', 'that', 'a', 'he', 'his']
open_class = ["be", "say", "come", "have", "see", "go", "know", "do", "levin", "would"]
for word in closed_class:
    print(word, " ", countdic[word], " ", get_n_similar(vecdic, 4, word))"""
"""print(get_most_common(30, countdic))
sent = "Alexey Alexandrovitch paused, and rubbed his forehead and his eyes. He saw that instead of doing as he had intended—that is to say, warning his wife against a mistake in the eyes of the world—he had unconsciously become agitated over what was the affair of her conscience, and was struggling against the barrier he fancied between them."
sentences = [sent.split()]
mat = frequency_matrix(2, 30, countdic, sentences)
vecdic = {}

i=0
for word in countdic:
    vecdic.update({word:mat[i]})
    i +=1

print(vecdic["warning"])"""
countdic = pickle.load(open("countdic3.1.p", "rb"))
vecdic = pickle.load(open("vecdic303.1.p", "rb"))
closed_class = ['the', 'of', 'in', 'and', 'not', 'to', 'that', 'a', 'he', 'his']
open_class = ['be', 'have','go',  'do', 'look', 'think', 'come', 'see', 'say', 'know']
"""singletons = get_n_singletons(countdic, 20)
for word in singletons:
    words, sim = get_n_similar(vecdic, 4, word)
    sort_similarity(word, words, sim)
    write_csv(countdic, word, words, sim, "singletons.csv" )"""
print(vecdic["his"])
