# -*- coding: utf-8 -*-
from __future__ import division
import sys,getopt,datetime,codecs
import csv
import re
from dateutil import parser
from datetime import timedelta
import datetime
from pytz import timezone
import pytz
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
import os, os.path
import chardet
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from itertools import chain
from imp import reload
import nltk
from nltk.wsd import lesk
import random
# STOP WORDS
from sklearn.feature_extraction.text import CountVectorizer
stopWords = set(CountVectorizer(stop_words='english').get_stop_words())



##############Tweet PreProcessor##############
def process(sentence):
	tokens = sentence.split() #StopWordRemoval
	sent2 = list()
	for word in tokens:
		if word.isupper() or word.lower() in stopWords:
			continue
		else:
			sent2.append(word)
	procSent = ' '.join( sent2 )
######other cleaning steps on data
	processed = procSent
	processed = re.sub('[^\x00-\x7f]*', '', processed)# to remove non-ascii characters
	processed = ' '.join([w for w in word_tokenize(processed)])
	processed = re.sub(regex, "", processed, 0) #to remove punctuations from string
	processed = processed.replace('?','')
	processed = processed.replace('\r','')
#	 if(expand):
#		 processed = _add_synset(processed)#in case we want to expand the text by wordnet
	return(processed)


##############Tweet PreProcessor##############
emoticons_str = r"""
	(?:
		[:=;] # Eyes
		[oO\-]? # Nose (optional)
		[D\)\]\(\]/\\OpP] # Mouth
	)"""

regex_str = [
	emoticons_str,
	r'<[^>]+>', # HTML tags
	r'(?:@[\w_]+)', # @-mentions
	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

	r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
	r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
	r'(?:[\w_]+)', # other words
	r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tweet_tokenize(s):
	return tokens_re.findall(s)

def tweet_preprocess(s, lowercase):
	tokens = tweet_tokenize(s)
	sent = list()
	if lowercase:
		tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
	
	for word in tokens:
		if ((word.isupper() and ('#' not in word)) or word.lower() in stopWords):
			continue
		else:
			sent.append(word)
	procSent = ' '.join( sent )
	procSent = re.sub(regex, "", procSent, 0)#to remove punctuations from string
	procSent = re.sub('[^\x00-\x7f]*', '', procSent)# to remove non-ascii characters
	return procSent


################# HashtagMatchCoverage #########
def HashtagMatchCoverage(tText,nText):
	HTMC = 0
	tTokens = tweet_tokenize(tText)
	HTs = list()
	for t in tTokens:
		if '#' in t:
			HTs.append(t)
	if len(HTs)==0:
		return HTMC
	else:
		nTokens = [w for w in word_tokenize(nText)]
		for ht in HTs:
			ht = ht.replace('#','')
			if ((ht.lower() in nTokens )or (ht.upper() in nTokens)):
				HTMC = HTMC + 1
		return (HTMC / (len(HTs)))
	


################# charNGramMatch ###########
def charNGramMatch(t, NA, n):
	tweetNgrams = [t[i:i+n] for i in range(len(t)-n+1)]
	NANgrams = [NA[i:i+n] for i in range(len(NA)-n+1)]
	MatchNgrams = list(set(tweetNgrams).intersection(NANgrams))
	MatchNgramsNum = len(MatchNgrams)
	TotalNgramPairs = len(NANgrams) * len(tweetNgrams)
	if TotalNgramPairs == 0:
		NGMatchedScore = 0
	else:
		NGMatchedScore = MatchNgramsNum / TotalNgramPairs
	return  NGMatchedScore




############### UTIL FUNCTIONS #########
def _get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN



##############_add_synset ############
from nltk.stem.snowball import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))
stemmed_nltk_stopwords = list(map(lambda w: snowball_stemmer.stem(w), nltk_stopwords))

def _add_synset(sentence):
	#sentence = sentence.decode('utf-8')
	tokens = sentence.split() # assume tokenized
	expansion = set()
	duplicate = list()
	for word, tag in nltk.pos_tag(tokens):
		if word.isupper() or word.lower() in stemmed_nltk_stopwords:
			continue
		synset = lesk(sentence, word, _get_wordnet_pos(tag))
		if synset:
			for lemma in synset.lemma_names():
				expansion.add(lemma) # includes word
		else:
			duplicate.append(word)
#if the original term should not be added, tokens must 
#be removed from the arguments of join() in the next line
	processed = ' '.join( tokens + duplicate + list(expansion) )
	return processed



################ CandidateNewsArticleFinder #########

def CandidateNewsArticleFinder(tIdNum,NotExp_TFIDFScores_Dir,Exp_TFIDFScores_Dir):
	TFIDFRetRes = {};#dictionary for each retrieved news article in the TFIDF res
	resFile = NotExp_TFIDFScores_Dir+"/"+str(tIdNum)+".txt"#location of the TFIDF score files
	file_open_read=open(resFile,'r')
	for rLine in file_open_read:
		rLine = rLine.strip()
		st = rLine.split(" ")
		newsId = st[2]
		TFIDFRetRes[newsId] = st[4] #scores and retrieved news ids
	file_open_read.close()
	Exp_TFIDFRetRes = {};#dictionary for each retrieved news article in the exp_TFIDF res
	exp_resFile = Exp_TFIDFScores_Dir+"/"+str(tIdNum)+".txt" #location of the TFIDF score files after expansion
	file_open_read=open(exp_resFile,'r')
	for rLine in file_open_read:
		rLine = rLine.strip()
		st = rLine.split(" ")
		newsId = st[2]
		Exp_TFIDFRetRes[newsId] = st[4] #scores and retrieved news ids
	file_open_read.close()

	commonNewsIds = ""
	for key in TFIDFRetRes:
		if key in Exp_TFIDFRetRes:
			commonNewsIds = commonNewsIds + key + ":" + TFIDFRetRes[key] + "," + Exp_TFIDFRetRes[key] + "|"
	return commonNewsIds



regex = r"(?<!\d)[.,;:](?!\d)"
def main(argv):
	paramFile = ''
	try:
		opts, args = getopt.getopt(argv, "hp:", ["pfile="])
	except getopt.GetoptError:
		print('Arguments parser error, try -h')
		return
	for opt, arg in opts:
		if opt == '-h':
			f = open('ReadMe.md', 'r')
			print(f.read())
			f.close()
			return
		elif opt in ("-p", "--pfile"):
			paramFile = arg

	outputFileName = "output_Linking.csv"
	with open(paramFile, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			tFile = row["TweetFile"]
			nFile = row["NewsArticlesFile"]
			lFile = row["LexiconFile"]
			mappingFile = row["MappingFile"]
			IdToDatemappingFile = row["IdToDateMappingFile"]
			outputFileName = row["output"]
			newsFileNum = row["nFileNum"]
			TFileNum = row["TweetNum"]
			NotExp_TFIDFScores_Dir = row["TFIDFScores_Dir"]
			Exp_TFIDFScores_Dir = row["Expanded_TFIDFScores_Dir"]

	try:
		print("tFile: "+str(tFile))
		print("nFile: "+str(nFile))
		print("IdToDateMappingFile: "+str(IdToDatemappingFile))
		print("outputFileName: "+str(outputFileName))
		print("TFileNum: "+str(TFileNum))
		print("Exp_TFIDFScores_Dir: "+str(Exp_TFIDFScores_Dir))
		#newsFileNum = 0
		outputFile = codecs.open(outputFileName, "w+", "utf-8")
		outputFile.write("tId;nId;c2gramMatchScore;c3gramMatchScore;Exp_c2gramMatchScore;Exp_c3gramMatchScore;diff;TFIDFScore;Exp_TFIDFScore;HashTagMC;IsMatched"+"\n")
		#news file id  to name mappings into a dict()
		mappingFileAddr = mappingFile
		nIdToFNameMapping = {}
		file_open_read3=open(mappingFileAddr,'r')
		for mLine in file_open_read3:
			ms = mLine.split(' ')
			nId = ms[0].strip()
			nFName = ms[1].strip()
			nIdToFNameMapping[nId] = nFName
		file_open_read3.close()
		
		#news file name  to Date mappings into a dict()		
		NFileDates = IdToDatemappingFile
		FNameToDateMapping = {}
		file_open_read3=open(NFileDates,'r')
		for mLine in file_open_read3:
			ms = mLine.rsplit('-',1)
			nDate = ms[0].strip()
			nFName = ms[1].strip()
			FNameToDateMapping[nFName] = nDate
		file_open_read3.close()
		
		#nId to nText mappings into a dict()
		NewFilesDir = nFile
		nIdToNText = {}
		for nId in range(0,int(newsFileNum)):	
			nFName = nIdToFNameMapping[str(nId)]
			file_open_read=open(NewFilesDir+"/"+str(nFName),'r')
			l = ""
			for nLine in file_open_read:
				nLine = nLine.strip()
				l = l + nLine + " "
			nIdToNText[nFName]=l
		#Read the Earthquake lexicon
		lexiconAddr = lFile
		earthquakeLexicon = []
		file_open_read4=open(lexiconAddr,'r')
		for mLine in file_open_read4:
			word = mLine.strip()
			earthquakeLexicon.append(word)
		file_open_read4.close()
		
	
		print('Searching...\n')
		cnt=0
		tIdNum = 0
		file_open_read1=open(tFile,'r')
		file_open_read1.readline()#skip the first line (headers)
		SkipStepSize = int(100 * random.uniform(0, 1))
		ContinueStepSize = int(5 * random.uniform(0, 1))
		for tLine in file_open_read1:
			print("tIdNum: "+str(tIdNum)+"\n")
			if(tIdNum<int(TFileNum)):
				tIdNum = tIdNum + 1
				continue

			#tweet line parsing
			tLine = tLine.strip()
			tweetFrags = tLine.split(";")
			tText = tweetFrags[4].replace('\r', '').split("http://")[0].strip()
			tText = tText.split("https://")[0]
			RtweetFrags = tLine.rsplit(";",2)#split the string from the right side into 2 splits, then read the second item
			tId = RtweetFrags[1].strip().replace('\r', '')
			tDate = tweetFrags[1]
			tId = tId.strip('\"')

			RefNID = "NotLabeled"
			tText = tweet_preprocess(tText.strip('\"'),False)# tweet specific preprocessor
			
			if any(term in tText for term in earthquakeLexicon): # if the tweet contains at least one word from the lexicon
				if SkipStepSize>0:
					tIdNum = tIdNum + 1
					SkipStepSize = SkipStepSize - 1
					continue
				elif ContinueStepSize>0:
					ContinueStepSize = ContinueStepSize - 1
					cnt = cnt + 1
					ExptText = _add_synset(tText)
					commonNewsIds = CandidateNewsArticleFinder(tId,NotExp_TFIDFScores_Dir,Exp_TFIDFScores_Dir)
					candNews = commonNewsIds.split("|")
					for NA in candNews:
						if len(NA)<2:
							continue
						else:
							nId = NA.split(":")[0]
							scores = NA.split(":")[1].split(",")
							#feature1 and feature 2: TFIDF Score & TFIDF Exp Score
							TFIDFScore = scores[0]
							Exp_TFIDFScore = scores[1]
							#mapping from nId to nText
							nFName = nIdToFNameMapping[str(nId)]
							nText = nIdToNText[nFName]
							nText = process(nText)
							ExpnText = _add_synset(nText)#in case we want to expand the text by wordnet
					
					
							#feature3: Char Ngram Match Score which is #MatchedNgrams/#TotalNgramPairs
							c2gramMatchScore = charNGramMatch(tText,nText,2)
							c3gramMatchScore = charNGramMatch(tText,nText,3)
					
					
							#feature4: Expanded Char Ngram Match Score which is #MatchedNgrams/#TotalNgramPairs
							Exp_c2gramMatchScore = charNGramMatch(ExptText,ExpnText,2)
							Exp_c3gramMatchScore = charNGramMatch(ExptText,ExpnText,3)

							#feature5: Temporal Distance
							####For tweet####
							#Example: 2015-04-29 18:59
							stdate1 = tDate.split(' ')
							tDate2 = stdate1[0]
							tdatetime_obj = datetime.strptime(tDate2, "%Y-%m-%d")


							#####for news####
							#Example: 2015-05-13
							nDate = FNameToDateMapping[nFName]
							

							ndatetime_obj = datetime.strptime(nDate, "%Y-%m-%d")

							#### Temporal difference ####
							diff = (ndatetime_obj - tdatetime_obj).days


							#feature6: Hashtag MatchCoverage
							HashTagMC = HashtagMatchCoverage(tText,nText)


							############write all features into file
							outputFile.write(str(tId)+","+str(nId)+","+str(c2gramMatchScore)+","+str(c3gramMatchScore)+","+str(Exp_c2gramMatchScore)+","+str(Exp_c3gramMatchScore)+","+str(diff)+","+str(TFIDFScore)+","+str(Exp_TFIDFScore)+","+str(HashTagMC)+","+RefNID+"\n")
							outputFile.flush()
				elif ContinueStepSize==0:
					SkipStepSize = int(100 * random.uniform(0, 1))
					ContinueStepSize = int(5 * random.uniform(0, 1))
			tIdNum = tIdNum + 1 # increment the tIdnum from 0 to <300K		
		outputFile.close()
	except:
		print('An exception occured!')
	finally:
		outputFile.close()
		print('Done. Output file generated "%s".' % outputFileName)

if __name__ == '__main__':
	main(sys.argv[1:])

