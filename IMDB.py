# Q1) Task 2
import requests  # import request to extract content from url
from bs4 import BeautifulSoup as bs  # BeautifulSoup is for web scraping used to scrap specific content
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty review list 
movie_review = []
  ip=[]  
  url="https://www.imdb.com/title/tt10083340/reviews?ref_=tt_urv"
  response = requests.get(url)
  response
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  soup
  reviews = soup.find_all("div", attrs={"class":"text show-more__control"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
movie_reviews = movie_review + ip   # adding the reviews of one page to empty list which in future contains all the reviews

# writing reviews in text file

 with open("gangubai.text","w",encoding = "utf8") as output:
     output.write(str(movie_reviews))
     
# joining all the reviews in a single paragraph

ip_rev_string = " ".join(movie_reviews)
   
from autocorrect import Speller
spell = Speller(lang = 'en')
ip_rev_string = spell(ip_rev_string)

import nltk 


# removing unwanted symbols incase if exist
ip_rev_string = re.sub("[^A_Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9""]+"," ",ip_rev_string)

# words that  contained in movie_reviews
ip_review_words = ip_rev_string.split(" ")


# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ip_review_words, use_idf= True, ngram_range=(1,3))
x = vectorizer.fit_transform(ip_review_words)  # converted into sparse matrix and it can't be readed

# custom stopwords

with open("C:\\Users\\praja\\Desktop\\Data Science\\Text mining sentiment analysis\\datasets\\stop.txt") as sw :
    stop_words = sw.read()       # importing stopwords
    
stop_words = stop_words.split("\n")

stop_words.extend(["watch","time", "angubai","athiawadi", "super", "movie","films","alia","bollywood"])

# filtering the data by removing stopwords

ip_review_words = [w for w in ip_review_words if not w in stop_words]

ip_rev_string = " ".join(ip_review_words)

# generating wprldcloud
# wordcloud can be performed on the string output

wordcloud_ip = WordCloud(
                         background_color = "white",
                         width = 1800,
                         height = 1400
                         ).generate(ip_rev_string)
plt.imshow(wordcloud_ip)
plt.title(" reviews of gangubai kathiawadi")
plt.axis("off")
plt.show()

# # positive words # Choose the path for +ve words stored in system
 with open("C:\\Users\\praja\\Desktop\\Data Science\\Text mining sentiment analysis\\datasets\\positive-words.txt") as pw:
      positivewords  = pw.read()

# postive wordcloud
# Choosing the only words which are present in positive words

ip_pos_in_pos = ' '.join([word for word in ip_review_words if word in positivewords])


wordcloud_pos = WordCloud(
                         background_color = 'White',
                         width = 1000,
                         height = 700,
                         max_words = 300
                         ).generate(ip_pos_in_pos)
plt.imshow(wordcloud_pos)
plt.title(" wordcloud for positive words")
plt.axis("off")

# negative wordcloud

with open("C:\\Users\\praja\\Desktop\\Data Science\\Text mining sentiment analysis\\datasets\\negative-words.txt") as nw:
    negativewords = nw.read()
    
ip_neg_neg = " ".join([w for w in ip_review_words if w in negativewords])

wordcloud_neg = WordCloud( background_color ='Black',
                          width = 1200, height = 800, 
                          max_words = 300).generate(ip_neg_neg)
plt.imshow(wordcloud_neg)
plt.axis("off")
plt.tiltle("wordcloud for negative words")
plt.show()

# wordcloud for bigram

nltk.download('punkt')
from wordcloud import WordCloud , STOPWORDS

WNL = nltk.WordNetLemmatizer()

# lowercase and tokenize 

text= ip_rev_string.lower()
# remove singlequote early since it causes problems with the tokenizer

 text = text.replace("'","")
 
 tokens = nltk.word_tokenize(text)
 text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.

text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords

stopwords_wc = set(STOPWORDS)
customized_words = [ "watch", 'alia','watch',"film","picture", " anjay", "eela","ussain","zaidi","hansali", "umbai","ueens","ahim","ala"]
new_words = stopwords_wc.union(customized_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_words]

# Take only non-empty entries

text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words

text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud( background_color= "White", max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_words)
wordCloud.generate_from_frequencies(words_dict)

plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


