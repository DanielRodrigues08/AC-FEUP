from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve


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


def train_model(model, df, year, target, param_grid):
    x_train, y_train, _, _ = split_data(df, year, target)

    grid_search = GridSearchCV(model, param_grid, cv=None)
    grid_search.fit(x_train, y_train)
    model.set_params(**grid_search.best_params_)

    model.fit(x_train, y_train)

    return model


def test_model(model, df, year, target):
    _, _, x_test, y_test = split_data(df, year, target)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return y_test, y_pred, y_prob


def plot_learning_curve(model, df, year):
    x_train, y_train, _, _ = split_data(df, year)
    train_sizes, train_scores, val_scores = learning_curve(model, x_train, y_train, cv=None)
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.legend(f"Learning Curve for year {year}")
    plt.show()
