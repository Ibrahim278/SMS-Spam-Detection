import numpy, pandas, re, os , glob, sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from bs4 import BeautifulSoup    
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
import matplotlib.pyplot as plt

df2=pandas.read_csv("spam.csv",sep=",")
#print df2
document=df2.iloc[:,1]
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, "html").get_text()   # 1. Remove HTML     
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) # 2. Remove non-letters
    words = letters_only.lower().split()         # 3. Convert to lower case, split into individual words  
    stops = set(stopwords.words("english"))          # 4. creating set of stop words       
    meaningful_words = [w for w in words if not w in stops] # 5. creating list of meaningful words by using loop that 
    #if words are not in stop set then they are meaningful
    return( " ".join( meaningful_words ))   # 6. Join the words back into one string separated by space, 
    # and return the result.

clean_train_reviews = []   # Initialize an empty list to hold the clean reviews
print "Cleaning and parsing the training set document...\n"
for i in xrange( 0, len(document)): 
    print i, len(document)                                                 
    clean_train_reviews.append( review_to_words( document[i] ))

vectorizer = TfidfVectorizer(stop_words=None,max_features = 1000)
train_data_features = vectorizer.fit_transform(clean_train_reviews) 

#HPL_dictionary =vectorizer.vocabulary_.items()
#tf=vectorizer.get_feature_names()
#print tf
#vocab = []
#count = []
# iterate through each vocab and count append the value to designated lists
#for key, value in HPL_dictionary:
    #vocab.append(key)
    #count.append(value)
# store the count in panadas dataframe with vocab as index    
#HPL_vocab = pandas.Series(count, index=vocab)
# sort the dataframe
#HPL_vocab = HPL_vocab.sort_values(ascending=False)
# plot of the top vocab
#top_vacab = HPL_vocab.head(20)
#top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (9300, 9330))
#plt.savefig("features.png")


X = train_data_features.toarray() # Numpy arrays are easy to work with, so convert the result to an
Y=df2.iloc[:,0]
from imblearn.over_sampling import RandomOverSampler,SMOTE

#ros = SMOTE()
#X,Y= ros.fit_sample(X, Y)
#print "after Smote",len(X)
trainx, testx, trainy, testy = train_test_split(X, Y,test_size=0.20, train_size=0.80, random_state=20000,shuffle=True)

#model= RandomForestClassifier(n_estimators = 600) 
#model = svm.SVC()
#model= LogisticRegression(C=0.01) 
model = GaussianNB()
#model=ensemble.GradientBoostingRegressor()


model = model.fit( trainx, trainy )
print sklearn.metrics.classification_report(testy,model.predict(testx))
print ("Final Accuracy: %s" % accuracy_score(testy, model.predict(testx)))
'''
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves

y_probas = model.predict_proba(testx)
skplt.metrics.plot_roc_curve(testy, y_probas,text_fontsize=8)
plt.savefig("roc_curve.png")
skplt.metrics.plot_precision_recall_curve(testy, y_probas,text_fontsize=8,title_fontsize=8)
plt.savefig("precision_recall.png")
plot_learning_curves(trainx, trainy, testx, testy, model,scoring='accuracy',print_model=False)
plt.savefig("training_test.png")
'''