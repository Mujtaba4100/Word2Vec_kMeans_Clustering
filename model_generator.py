import pandas as pd 
import numpy as np
import gensim
from gensim.models import Word2Vec


df=pd.read_csv('reddit_worldnews_start_to_2016-11-22.csv')
df.shape
newsTitles=df['title'].values
# print(len(newsTitles)) 

newsTitles=df['title'].apply(gensim.utils.simple_preprocess)
newsTitles.shape

model = Word2Vec(min_count=20, window=2,alpha=0.03,min_alpha=0.0007, workers=6)
model.build_vocab(newsTitles,progress_per=1000)
epochs=model.epochs
model.train(newsTitles, total_examples=model.corpus_count, epochs=epochs)
model.save("Word2Vec.model")
print("Model Trained successfully")

data = []

for word in model.wv.index_to_key:
    vector = model.wv[word]
    data.append([word] + list(vector))

df = pd.DataFrame(data)

df.columns = ['Word'] + [f'Vec_{i+1}' for i in range(100)]

df.to_csv('word_vectors.csv', index=False)

print("Word vectors stored to 'word_vectors.csv'")
