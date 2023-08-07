## Movie Recommendation System Frontend (Streamlit)
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import import_ipynb
from model_2 import moviemat
recommender = pickle.load(open('model.pkl','rb'))
def recommend2(movieName):
    distances, indices = recommender.kneighbors(moviemat.loc[movieName,:].values.reshape(1,-1),n_neighbors=7)
    movie_list = {}
    print(distances)
    print(indices)
    d = distances.flatten()
    for i in range(0,len(d)):
        if i > 0:
            ind = moviemat.index[indices.flatten()]
            # print("{0}: {1} with distance: {2}\n".format(i,ind[i],d[i]))
            movie_list[ind[i]] = d[i]
    df = pd.DataFrame(movie_list.items(), columns=['Movie Name','Closeness'])
    return df
with open('D:\F\Coding\Projects\Movie Recommendation\style.css') as f:
    st.write(f'<style>{f.read()}</style>',unsafe_allow_html=True)
st.markdown("""# MOVIE RECOMMENDATION SYSTEM""")
st.subheader("Welcome to AI based online Movie Recommendation System")
txt = st.text_input("Enter movie name: ")
df = recommend2(txt)
st.text("Here is the list is the list of all the recommended movies similar to {txt}: ")
st.table(df)