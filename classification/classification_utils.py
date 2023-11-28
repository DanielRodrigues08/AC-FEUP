from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve
import pandas as pd
import copy


def split_by_conf(df):
    return df[df["confID"] == 0], df[df["confID"] == 1]


def split_data(df, year, target):
    
    train_data = df[df["year"] < year]
    test_data = df[df["year"] == year]

    x_train = train_data.drop([target], axis=1)
    y_train = train_data[target]

    x_test = test_data.drop([target], axis=1)
    y_test = test_data[target]

    return x_train, y_train, x_test, y_test


def train_model_simple(classifier, df, year, target):
    x_train, y_train, _, _ = split_data(df, year, target)
    df['sampleWeight'] = df['year'].apply(lambda year_x: 2 ** (year - year_x - 1) if year > year_x else 1)
    try:
        classifier.fit(x_train, y_train, sample_weight=df.loc[x_train.index]['sampleWeight'])
    except:
        for i, (_, row) in enumerate(x_train.iterrows()):
            sample_weight = df.loc[row.name]['sampleWeight']
            classifier.partial_fit([row.values], [y_train.iloc[i]], sample_weight=[sample_weight])


def train_model_hyper_tunning(classifier, df, year, target, param_grid):
    x_train, y_train, _, _ = split_data(df, year, target)

    grid_search = GridSearchCV(classifier, param_grid, cv=None)
    grid_search.fit(x_train, y_train)
    classifier.set_params(**grid_search.best_params_)
    classifier.fit(x_train, y_train)

def test_model(model, df, year, target):
    x_train, y_train, x_test, y_test = split_data(df, year, target)

    y_test_prob = model.predict_proba(x_test)[:, 1]
    y_train_prob = model.predict_proba(x_train)[:, 1]

    return y_test, y_test_prob, x_test["confID"], y_train, y_train_prob, x_train["confID"]


def enforce_max_teams(y_prob, conf_id, max_teams=4):
    joined = zip(range(len(y_prob)), y_prob, conf_id)
    joined = sorted(joined, key=lambda x: x[1], reverse=True)

    y_pred = [0 for _ in range(len(y_prob))]

    count_0 = 0
    count_1 = 0
    for i, _, conf in joined:
        if count_0 < max_teams and conf == 0:
            y_pred[i] = 1
            count_0 += 1
        elif count_1 < max_teams and conf == 1:
            y_pred[i] = 1
            count_1 += 1
        else:
            continue

    return y_pred
