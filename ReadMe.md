* The implementation of a tweet-news article linking method proposed in the following article:
	- Verma, Rakesh, Samaneh Karimi, Daniel D. Lee, Omprakash Gnawali, and Azadeh Shakery. “Newswire versus Social Media for Disaster Response and Recovery.” In 2019 Resilience Week (RWS), San Antonio, TX, USA, pp. 132-141. 2019.

* To use this script you should pass the following parameters in a param.csv file:
	- TweetFile: The location of the tweet file.
	- NewsArticlesFile: The location of the News file.
	- LexiconFile: The location of a lexicon file containing words related to the tweets and news topic (For more details refer to the paper)
	- MappingFile: The mapping between DocIds used by the Retrieval Tool and their Names
	- IdToDateMappingFile: The location of the file containing news articles dates 
	- output: A filename to export the results (default is "output_Linking.csv")
	- nFileNum: The number of news documents in the dataset
	- TweetNum: The number of tweets in the dataset
	- TFIDFScores_Dir: The directory containing TFIDF scores of all documents
	- Expanded_TFIDFScores_Dir: The directory containing TFIDF scores of all documents expanded using wordNet as explained in the paper


* How to Run the manuscript:
	- You can run this manuscript in Python3 using the following command:
		python3 TNLinking.py —p param.csv
	- You can see the ReadMe.md file using -h option:
		python3 TNLinking.py —h

* Citation 
	- Verma, Rakesh, Samaneh Karimi, Daniel D. Lee, Omprakash Gnawali, and Azadeh Shakery. “Newswire versus Social Media for Disaster Response and Recovery.” In 2019 Resilience Week (RWS), San Antonio, TX, USA, pp. 132-141. 2019.
