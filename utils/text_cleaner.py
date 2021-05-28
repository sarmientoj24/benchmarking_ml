from nltk import word_tokenize
from sklearn.base import TransformerMixin
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()
punctuations = string.punctuation
stop_words = set([i.lower() for i in stopwords.words('english')]) 


# Custom transformer using spaCy
class TextCleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [self.clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    # Creating our tokenizer function
    def tokenizer(self, sentence):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = word_tokenize(sentence)

        # Lemmatizing each token and converting each token into lowercase
        mytokens = [re.sub("[^a-zA-Z0-9]", "", word.strip()) for word in mytokens]
        
        mytokens = [stemmer.stem(word) for word in mytokens if word]

        # Removing stop words
        mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

        # return preprocessed list of tokens
        return mytokens

    # Basic function to clean the text
    def clean_text(self, text):
        tokens = self.tokenizer(text.strip().lower())

        # Removing spaces and converting text into lowercase
        return ' '.join(tokens)
