import numpy as np
import pandas as pd

def recommend(x, y):
    # Cosinus Probability with every other user
    z = np.dot(np.nan_to_num(x / np.nan_to_num(np.linalg.norm(x, axis=0))),
                y / np.linalg.norm(y))

    # Calc Normalized Matrix
    X = np.nan_to_num(x / np.nan_to_num(np.linalg.norm(x, axis=0)))

    # Normalize Vector to get vid-profile representation
    Z = z / np.linalg.norm(z)

    return np.dot(X.T, Z)

def get_users_ratings(data_r, data_m):

    # Maximum Movie Length
    movie_len = max(set(data_r['movieId'])) + 1

    # Temporary IDs
    users = {x: i for i, x in enumerate(set(data_r['userId']))}
    ratings = np.zeros((len(users), movie_len))

    # Retrieve User Ratings
    for row in data_r.itertuples():
        ratings[users[int(row.userId)], int(row.movieId)] = row.rating
    return ratings

def main():
    # Retrieve 10000 Records of data
    data_r = pd.read_csv("./ratings.csv", sep=",")
    data_m = pd.read_csv('./movies.csv', sep=",")
    data_r = data_r.loc[data_r["movieId"] < 10000]
    movies = data_m[(data_m.movieId < 10000)]

    # Define custom ratings for videos to get your recommendations
    my_ratings = np.zeros((9019, 1))
    my_ratings[2571] = 5
    my_ratings[32] = 4
    my_ratings[260] = 5
    my_ratings[1097] = 4

    # Getting All Ratings
    users_ratings = get_users_ratings(data_r, data_m)
    recommended = recommend(users_ratings, my_ratings)

    # to_recommend = best(recommended, movies)
    best = sorted(((x[0], movies[(movies['movieId'] == i)])
                  for i, x in enumerate(recommended)),
                  key=lambda x: x[0], reverse=True)
        
    for i, x in enumerate(best[:5]):
        print("------ \n# Rank: (", i+1, ")\n Score: ", x[0],
              "\n# Film Info: \n", x[1])

if __name__ == "__main__":
    main()

# Legacy

# Sample Input
# x = np.array([[5, 0, 2, 5], [4, 1, 2, 5]]).T
# y = np.array([[4], [3]])
# recommended = recommend(x, y)

# def best(recommended, movies):
#     to_recommend = []
#     for i, x in enumerate(recommended):
#         print(x[0])
#         if x > 0.01:
#             mov = [x[0], movies[(movies['movieId'] == i)].title]
#             to_recommend.append(mov)
#     for a in to_recommend:
#         print(a)
#     # Python sorts nested arrays by first element
#     sorted(to_recommend)
#     return to_recommend
