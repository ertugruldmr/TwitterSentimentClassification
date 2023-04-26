import re
import pickle
import numpy as np
import gradio as gr
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# File Paths
model_path = 'loj_reg_twitter_sentiment.sav'
bow_vectorizer_path = "bow_vectorizer.sav"

# Loading the files
model = pickle.load(open(model_path, 'rb'))
bow_vectorizer = pickle.load(open(bow_vectorizer_path, 'rb'))
stemmer = PorterStemmer()

labels = ["negative", "positive"]#classes[target].values()

Examples = [
    "Very bad, worst",
    "perfect, very good",
    "I just had the best meal at my favorite restaurant. The food was delicious and the service was fantastic!",
    " I'm so disappointed with the customer service I received from this company. They were unhelpful and rude, and I won't be using their services again."
]


# Load the model
def text_preprocessing(input_txt, pattern:str="@[\w]*"):
  # Finding all the texts which fits the pattern
  r = re.findall(pattern, input_txt)
  
  # removing this words
  for word in r: input_txt = re.sub(word, "", input_txt)

  #  removing special characters
  input_txt = input_txt.replace("[^a-zA-Z#]", " ")
  
  # standart lowercase
  input_txt = str.lower(input_txt)

  # tokenization
  tokens = input_txt.split()

  # stemming for standardization
  tokens = [stemmer.stem(word) for word in tokens]

  # concatenating the words
  sentence = " ".join(tokens)

  return sentence

def vectorizer(sentence):
   return bow_vectorizer.transform(sentence)
  
def predict(text):

  # preparing the input into convenient form
  sentence = text_preprocessing(text)
  
  # vectorizing the data  
  features = vectorizer([sentence])

  # prediction
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results

# GUI Component
demo = gr.Interface(predict, "text", "label", examples = Examples)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
