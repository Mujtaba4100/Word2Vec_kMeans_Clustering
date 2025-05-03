import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import tkinter as tk
from tkinter import simpledialog, messagebox
from kmeans import KMeans

model = Word2Vec.load("Word2Vec.model")

df = pd.read_csv('word_vectors.csv')
X = df.iloc[:, 1:].values  


# y_means = np.load('kmeans_labels.npy')

def perform_clustering(X, cluster_sizes):
   
    cluster_labels = {}

    
    for n_clusters in cluster_sizes:
        km = KMeans(n_clusters=n_clusters)
        y_means = km.fit_predict(X)
        cluster_labels[n_clusters] = y_means
        print(f"Cluster labels for {n_clusters} clusters: {np.unique(y_means)}")
    print("Clustering completed")
    return cluster_labels

def plot_3d(X, cluster_labels, cluster_sizes):

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    df_reduced = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3'])

    for n_clusters in cluster_sizes:
        y_means = cluster_labels[n_clusters]
        fig = px.scatter_3d(df_reduced, x='PC1', y='PC2', z='PC3', color=y_means,
                            title=f"3D Plot of {n_clusters} Clusters (100D Word Vectors)")
        fig.show()


def plot_2d():
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    df_reduced = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
    
    plt.scatter(df_reduced['PC1'], df_reduced['PC2'], c=y_means, cmap='viridis')
    plt.title("2D Plot of 100D Word Vectors")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Cluster')
    plt.show()

def most_similar_word():
    word = simpledialog.askstring("Input", "Enter a word to find its most similar word:")
    if word in model.wv:
        similar = model.wv.most_similar(word)
        result = "\n".join([f"{w[0]}: {w[1]}" for w in similar])
        messagebox.showinfo("Most Similar Words", result)
    else:
        messagebox.showerror("Error", "Word not found in the model vocabulary.")

def check_similarity():
    word1 = simpledialog.askstring("Input", "Enter the first word:")
    word2 = simpledialog.askstring("Input", "Enter the second word:")
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2)
        messagebox.showinfo("Word Similarity", f"Similarity between '{word1}' and '{word2}': {similarity}")
    else:
        messagebox.showerror("Error", "One or both words not found in the model vocabulary.")

def odd_one_out():
    words = simpledialog.askstring("Input", "Enter three words separated by commas (e.g., 'bad, good, apple'):")
    words = words.split(',')
    if all(word in model.wv for word in words):
        odd_word = model.wv.doesnt_match(words)
        messagebox.showinfo("Odd One Out", f"The odd one out is: {odd_word}")
    else:
        messagebox.showerror("Error", "One or more words not found in the model vocabulary.")

root = tk.Tk()
root.title("Word2Vec GUI")
root.geometry("400x300")


def on_option_select():
    option = simpledialog.askstring("Choose an option", 
                                    "1.Enter CLuster sizes\n2. Most Similar Word\n3. Word Similarity\n4. Odd One Out\n5. 2D Plot\n6. 3D Plot\nSelect the number of the option:")
    if option =='1':
        print("")
        cluster_sizes = simpledialog.askstring(" Input", "Larger Cluster size might take longer time to compute\n Enter cluster sizes separated by commas (e.g., 2,5,7,9):")
        cluster_sizes = [int(x.strip()) for x in cluster_sizes.split(',')]#to convert string into integers
        cluster_labels=perform_clustering(X, cluster_sizes)
        plot_3d(X,cluster_labels,cluster_sizes)
    elif option == '2':
        most_similar_word()
    elif option == '3':
        check_similarity()
    elif option == '4':
        odd_one_out()
    elif option == '5':
        plot_2d()
    elif option == '6':
        print("3D plot is already handled in option 1.")
    else:
        messagebox.showerror("Invalid Option", "Please select a valid option.")

btn = tk.Button(root, text="Choose Option", command=on_option_select)
btn.pack(pady=20)

root.mainloop()

