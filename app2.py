from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = pickle.load(open(MODEL_PATH,'rb'))

@app.route("/",methods = ['GET','POST'])
@app.route("/home",methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
    return render_template("index.html")

@app.route("/recommend",methods = ['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        movieName = request.form["moviename"]
        with open('moviemat.pkl', 'rb') as file:
            moviemat = pickle.load(file)

        distances,indices = model.kneighbors(moviemat.loc[movieName,:].values.reshape(1,-1),n_neighbors = 5)
        movie_list = {}
        d = distances.flatten()
        for i in range(0,len(d)):
            if i > 0:
                ind = moviemat.index[indices.flatten()]
                movie_list[ind[i]] = d[i]
            df = pd.DataFrame(movie_list.items(), columns=['Movie Name','KNN Distance'])
            print(df)
            return render_template('index.html',tables = df.to_html(), titles = df.columns.values)







@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug = True, port=5500)