from django.http import HttpResponse
from django.shortcuts import render
import operator
import string
import re

def home(request):
    return render(request, 'home.html')

def count(request):
    fullText = request.GET['fullText']
    sentenceslist = re.split(r'[.!?]+', fullText)

    lowerFullText = fullText.lower()
    newFullText = ""

    for i in lowerFullText:
        if i not in string.punctuation:
            newFullText += i

    wordlist = newFullText.split()

    wordDictionary = {}

    for word in wordlist:
        if word in wordDictionary:
            #Increase
            wordDictionary[word] += 1
        else:
            #Add to the dictionary
            wordDictionary[word] = 1
    sort_wordDictionary = sorted(wordDictionary.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return render(request, 'count.html', {'fullText':fullText, 'count':len(wordlist), 'wordDictionary': sort_wordDictionary, 'sentencesCount':len(sentenceslist)-1})