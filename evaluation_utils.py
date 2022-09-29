import numpy as np
from hyperopt.pyll import scope
from hyperopt import hp
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector


seed = 1995


def get_domain_dict():
    target_dict = {
        "target": {
            "target_problem": "target-prediction",
            "TARGET": "column_target"
        }
    }
    return target_dict


def get_selected_features(df, TARGET):
    init_features = [x for x in list(df.columns) if "yeaaaah" not in x]
    X = df[init_features].copy()  # Feature Matrix
    y = df[TARGET].copy()  # Target Variable
    X.fillna(0, inplace=True)
    y = y*100  # upscaling small values (percentages)
    for col in init_features:
        if 'pct_' in col:
            X[col] = X[col]*100  # upscaling small values (percentages)
    standard_scaler = StandardScaler()
    pipeline = Pipeline(
        [('transformer', standard_scaler), ('estimator', XGBRegressor())])
    efs = SequentialFeatureSelector(pipeline,
                                    forward=False,
                                    scoring='neg_mean_squared_error',
                                    k_features=(5, 20),
                                    floating=False,
                                    verbose=0,
                                    cv=0)
    efs.fit(X, y)
    return efs.k_feature_names_


def get_filtered_features():
    features = []
    return features


def get_search_space_hyperopt():
    search_estimators = {
        "xgboost": {
            "model": XGBRegressor,
            "skip": False,
            "reduce": np.mean,
            "space": {
                # Reduced searh espace for n_estimators to avoid overfitting
                # on small datasets. Normally, the range would be 1-4000.
                # Check more ranges here:
                # https://leehanchung.github.io/2021-06-25-xgboost-hyperparameter-tuning/
                'n_estimators': scope.int(hp.uniform('n_estimators', 1, 100)),
                'eta': hp.loguniform('eta', -7, 0),
                'max_depth': scope.int(hp.uniform('max_depth', 1, 10)),
                'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
                'subsample': hp.uniform('subsample', 0.2, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                'gamma': hp.choice('gamma',
                                   [0, hp.loguniform('gamma1', -16, 2)]),
                'alpha': hp.choice('alpha',
                                   [0, hp.loguniform('alpha1', -16, 2)]),
                'lambda': hp.choice('lambda',
                                    [0, hp.loguniform('lambda1', -16, 2)]),
                'seed': seed,
            }
        },
        "randomForest": {
            "model": RandomForestRegressor,
            "skip": True,
            "reduce": np.mean,
            "space": {
                'n_estimators': scope.int(hp.randint('n_estimators',
                                                     50, 1000)),
                'max_depth': scope.int(hp.randint('max_depth', 5, 20)),
                'min_samples_leaf': scope.int(hp.randint('min_samples_leaf',
                                                         1, 5)),
                'min_samples_split': scope.int(hp.randint('min_samples_split',
                                                          2, 6))
            }
        },
        "ridge": {
            "model": Ridge,
            "skip": True,
            "reduce": np.mean,
            "space": {
                "alpha": hp.loguniform('alpha', -10, 10),
                "fit_intercept": hp.choice("fit_intercept", [True, False]),
                "solver": hp.choice("solver",
                                    ["svd", "cholesky", "lsqr", "sag", "saga"])
            }
        },
        "lasso": {
            "model": Lasso,
            "skip": True,
            "reduce": np.mean,
            "space": {
                "alpha": hp.uniform('alpha', 0, 10),
                "fit_intercept": hp.choice("fit_intercept", [True, False])
            }
        }
    }
    return search_estimators


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


def get_mape(estimator, x, y):
    yPred = estimator.predict(x)
    return (MAPE(y, yPred))


def mape_scorer(estimator, x, y):
    s = get_mape(estimator, x, y)
    # print(s)
    return s
