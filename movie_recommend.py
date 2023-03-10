import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

df = pd.read_csv("movies.csv")

features = ['genres','overview','production_companies','cast','crew','director']
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['genres']+" "+row['overview']+" "+row['production_companies']+" "+row['cast']+" "+row['crew']+" "+row['director']
    except:
        print ("error",row)

df['cf'] = df.apply(combine_features,axis = 1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['cf'])
sim = cosine_similarity(count_matrix)

fav_movie = "Inception"
fav_movie_index = get_index_from_title(fav_movie)

similar_movies = list(enumerate(sim[fav_movie_index]))

sorted_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True)

i = 0
for movie in sorted_movies:
    print (get_title_from_index(movie[0]))
    i+=1
    if i > 50:
        break
