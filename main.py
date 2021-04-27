import nltk
import matplotlib.pyplot as plt

# Sentence Tokenization
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
text="""Hello student. Welcome to the session on Python today. Having learnt pygames, tkinter, turtle graphics and OOPS, it is time for you to get introduced to plotting graphs and natural language processing"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

# Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

# Most common word
fdist.most_common(2)

fdist.plot(30,cumulative=False)
plt.show()

