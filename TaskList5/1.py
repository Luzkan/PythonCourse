import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def format_x(max_m, usr_ids):
    m_id = data_m['movieId'].to_numpy()
    idx = m_id[m_id <= max_m]
    r = data_r[(data_r['movieId'] <= max_m) &
               (data_r['userId'].isin(usr_ids))]
    x = r.pivot(index='userId', columns='movieId',
                values='rating').reindex(idx[1:], axis='columns')
    np.nan_to_num(x, copy=False)
    return x

# Graph: [x: m, all_users: Score]
def graph(scores):
    plt.plot(ms, scores, linewidth=5)
    plt.xscale('log')
    plt.xlabel('m')
    plt.ylabel('Score %')
    plt.grid()
    plt.show()

def get_scores(ms):
    scores = []
    for m in ms:
        x = format_x(m, notoy_users)
        clf = LinearRegression().fit(x, all_users)
        scores.append(clf.score(x, all_users))
    return scores

def res(p):
    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.scatter(np.arange(1, 16), p, c='blue', s=50, label='Prediction')
    ax.scatter(np.arange(1, 16), all_users[-15:], c='red', s=50,
               label='Expectation')
    ax.set_title('Results based on m = 10000')
    fig.text(0.5, 0.01, 'User', ha='center', va='center')
    fig.text(0.01, 0.5, 'Rating', ha='center', va='center',
             rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.show()

def learn(ms, predict_last=15):
    for m in ms:
        # Based on all w/o last 15 (out of 215) users
        x = format_x(m, notoy_users[:-predict_last])
        clf = LinearRegression().fit(x, all_users[:-predict_last])

        # Last 15 Users to be predicted
        last_15 = format_x(m, notoy_users[-predict_last:])
        p = clf.predict(last_15)
    return p

def do_lr(ms, predict_last):
    graph(get_scores(ms))
    prediction = learn(ms, predict_last)
    res(prediction)

if __name__ == "__main__":
    ms = [10, 100, 200, 500, 1000, 2500, 5000, 7500, 10000]
    data_r = pd.read_csv("./ratings.csv", sep=",")
    data_m = pd.read_csv('./movies.csv', sep=",")

    # There are 215 users who rated toystory
    users_r_toystory = data_r[data_r['movieId'] == 1][[
                       'rating', 'userId']].to_numpy()

    # Users indexed from 0 (w/ toystory) and 1 (w/o toystory)
    all_users = users_r_toystory[:, 0]
    notoy_users = users_r_toystory[:, 1]

    # Second args equal to last X users to be predicted
    # (reduces training pool)
    do_lr(ms, 15)