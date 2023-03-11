from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from predict_fed.pipeline import Pipeline


def tune_model(model, y, features, grid, search='random', pipe_params={}, train_size=0.8):
    pipe = Pipeline(y=y, features=features, split_percentages=(100 * train_size, 0, 100 * (1 - train_size)),
                    **pipe_params)

    X_train, X_valid, X_test, y_train, y_valid, y_test = pipe.run()

    m = model()
    m.init_model()
    regressor = m.model

    if search == 'random':
        searcher = RandomizedSearchCV(estimator=regressor, scoring='neg_mean_squared_error', n_jobs=-1, cv=5,
                                      param_distributions=grid, verbose=1)
    elif search == 'grid':
        searcher = GridSearchCV(estimator=regressor, scoring='neg_mean_squared_error', n_jobs=-1, cv=5,
                                param_grid=grid, verbose=1)
    else:
        raise Exception('Invalid search')

    searchResults = searcher.fit(X_train, y_train)

    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("Best score is {:.2f} using {}".format(bestScore, bestParams))
    return bestParams
