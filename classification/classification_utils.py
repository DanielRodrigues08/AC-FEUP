from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve
import pandas as pd
import copy


def split_by_conf(df):
    return df[df['confID'] == 0], df[df['confID'] == 1]


def split_data(df, year, target):
    train_data = df[df["year"] < year]
    test_data = df[df["year"] == year]

    x_train = train_data.drop([target], axis=1)
    y_train = train_data[target]

    x_test = test_data.drop([target], axis=1)
    y_test = test_data[target]

    return x_train, y_train, x_test, y_test


def train_model(classifier, df, year, target, param_grid):
    classifier1 = copy.deepcopy(classifier)  # to create a pure function
    x_train, y_train, _, _ = split_data(df, year, target)

    grid_search = GridSearchCV(classifier1, param_grid, cv=None)
    grid_search.fit(x_train, y_train)
    classifier1.set_params(**grid_search.best_params_)

    model = copy.deepcopy(classifier1)  # to enable plotting a learning curve with the same parameters
    model.fit(x_train, y_train)

    return model, classifier1


def test_model(model, df, year, target):
    _, _, x_test, y_test = split_data(df, year, target)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return y_test, y_pred, y_prob, x_test['confID']


def plot_learning_curve(classifier, df, year, target):
    classifier1 = copy.deepcopy(classifier)  # to create a pure function
    x_train, y_train, _, _ = split_data(df, year, target)
    train_sizes, train_scores, val_scores = learning_curve(classifier1, x_train, y_train, cv=None)
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.legend(f"Learning Curve for year {year}")
    plt.show()


def enforce_max_teams(y_pred, y_prob, conf_id, max_teams=4):
    joined = zip(range(len(y_pred)), y_prob, conf_id)
    joined = sorted(joined, key=lambda x: x[1], reverse=True)

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
            y_pred[i] = 0

    return y_pred
