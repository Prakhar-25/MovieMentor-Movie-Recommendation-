from flask import Flask, render_template, request,flash
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

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

        if movieName not in moviemat.index:
            # pass message to frontend
            flash('Sorry, no recommendations. Please try some other movie.','error')
            return render_template('index.html')

        distances,indices = model.kneighbors(moviemat.loc[movieName,:].values.reshape(1,-1),n_neighbors = 7)
        movie_list = {}
        # print(distances)
        # print(indices)
        d = distances.flatten()
        indices = indices.flatten()
        # print("moviemat.index: ", moviemat.index)
        for i in range(0,len(d)):
            if i > 0:
                ind = moviemat.index[indices.flatten()]
                # print("{0}: {1} with distance: {2}\n".format(i,ind[i],d[i]))
                movie_list[ind[i]] = d[i]
        df = pd.DataFrame(movie_list.items(), columns=['Movie Name','Closeness (Based on Ratings)'])
        # print(df)
        return render_template('index.html',tables = df.values.tolist(), titles = df.columns.values,result = True)







@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug = True, port=5500)