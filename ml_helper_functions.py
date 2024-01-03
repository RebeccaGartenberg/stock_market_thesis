from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import svm
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import dataframe_image as dfi
from statistics import mean

def normalize_data(training_data, labels):
    # All features are normalized to be between 0 and 1
    X = training_data.drop(columns=labels)
    profit_features = (['baseline_profits','sma_profits','sma_hourly_profits', 'stoch_profits', 'stoch_hourly_profits',
    'mean_rever_profits', 'mean_rever_hourly_profits', 'rsi_profits', 'rsi_hourly_profits']) # include low and high? can't include mean but then it doesn't make sense
    exchange_cols = X.filter(regex=f'^exchange_', axis=1).columns
    state_cols = X.filter(regex=f'^state_', axis=1).columns
    country_cols = X.filter(regex=f'^country_', axis=1).columns
    sector_cols = X.filter(regex=f'^sector_', axis=1).columns
    industry_cols = X.filter(regex=f'^industry_', axis=1).columns
    categorical_features = state_cols.tolist()+exchange_cols.tolist()+country_cols.tolist()+sector_cols.tolist()+industry_cols.tolist()+['gives_dividend']

    scaler_vertical = MinMaxScaler()

    X_categorical = X[categorical_features]
    X[profit_features] = X[profit_features].div(X['mean_price'], axis=0)
    X = X.drop(columns=categorical_features, axis=1)
    X_scaled_normalized = pd.DataFrame(scaler_vertical.fit_transform(X), columns=X.columns, index=X.index)
    X_test = pd.concat([X_scaled_normalized, X_categorical], axis=1)
    return pd.concat([training_data[labels], X_test], axis=1)

def grid_search(X_train, Y_train, labels):
    regression_models = []
    random_forest_models = []
    svm_models = []

    # Specified each combination because lbfgs cannot be paired with L1 norm and liblinear cannot be paired with penalty None
    regression_model_param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [0.1]},
        {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [1.0]},
        {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [10.0]},
        {'solver': ['liblinear'], 'penalty': ['l2'], 'C': [0.1]},
        {'solver': ['liblinear'], 'penalty': ['l2'], 'C': [1.0]},
        {'solver': ['liblinear'], 'penalty': ['l2'], 'C': [10.0]},
        {'solver': ['lbfgs'], 'penalty': [None], 'C': [0.1]},
        {'solver': ['lbfgs'], 'penalty': [None], 'C': [1.0]},
        {'solver': ['lbfgs'], 'penalty': [None], 'C': [10.0]},
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.1]},
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [1.0]},
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [10.0]}
    ]

    random_forest_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'], #'linear',
        'degree': [2, 3, 4]
    }

    start = time.time()
    for i in range(0, Y_train.shape[1]):
        # Linear Regression
        print(f'Training logistic regression model for label {i+1}/{Y_train.shape[1]}')
        regression_model = LogisticRegression(max_iter=1000)
        regression_model_grid_search = GridSearchCV(estimator=regression_model, param_grid=regression_model_param_grid, cv=3, scoring='accuracy')
        regression_model_grid_search.fit(X_train, Y_train[labels[i]])
        regression_models.append(regression_model_grid_search.best_estimator_)

        # Random Forest
        print(f'Training random forest for label {i+1}/{Y_train.shape[1]}')
        random_forest_model = RandomForestClassifier()
        random_forest_grid_search = GridSearchCV(estimator=random_forest_model, param_grid=random_forest_param_grid, cv=3, scoring='accuracy')
        random_forest_grid_search.fit(X_train, Y_train[labels[i]])
        random_forest_models.append(random_forest_grid_search.best_estimator_)

        # Support Vector Machine
        print(f'Training SVM for label {i+1}/{Y_train.shape[1]}')
        svm_model = svm.SVC()
        svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=3, scoring='accuracy')
        svm_grid_search.fit(X_train, Y_train[labels[i]])
        svm_models.append(svm_grid_search.best_estimator_)
    end = time.time()
    print(f'Total ML training time: {end-start}')

    return regression_models, random_forest_models, svm_models

def save_model_params(labels, regression_models, random_forest_models, svm_models, tables_dir_name, file_name):
    # Save models with best params here
    params = pd.DataFrame(columns=['regression', 'random forest', 'svm'])
    params['labels'] = labels
    params['regression'] = regression_models
    params['random forest'] = random_forest_models
    params['svm'] = svm_models
    params.to_csv(f'{tables_dir_name}/hyperparameters_{file_name}.csv', mode='a', header=True, index=False)
    return

def get_model_scores(Y_test, y_pred):
    accuracies_is_prof = [accuracy_score(Y_test.to_numpy()[:,i].tolist(), y_pred[:,i].tolist()) for i in range(Y_test.shape[1]-1)]
    accuracies_best_strat = accuracy_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred[:,Y_test.shape[1]-1])
    precisions_is_prof = [precision_score(Y_test.to_numpy()[:,i].tolist(), y_pred[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
    precisions_best_strat = precision_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
    recalls_is_prof = [recall_score(Y_test.to_numpy()[:,i].tolist(), y_pred[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
    recalls_best_strat = recall_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred[:,Y_test.shape[1]-1], average='macro', zero_division='warn')
    f1_is_prof = [f1_score(Y_test.to_numpy()[:,i].tolist(), y_pred[:,i].tolist(), average='binary', zero_division='warn') for i in range(Y_test.shape[1]-1)]
    f1_best_strat = f1_score(Y_test.to_numpy()[:,Y_test.shape[1]-1], y_pred[:,Y_test.shape[1]-1], average='macro', zero_division='warn')

    return accuracies_is_prof, accuracies_best_strat, precisions_is_prof, precisions_best_strat, recalls_is_prof, recalls_best_strat, f1_is_prof, f1_best_strat

def make_predictions(X_test, models):
    y_pred = np.column_stack([model.predict(X_test) for model in models])
    return y_pred

def create_and_save_table(scores, labels, tables_dir_name, file_name):
    results = ({
            'Label' : labels,
            'Accuracy': scores[0] + [scores[1]],
            'Precision': scores[2] + [scores[3]],
            'Recall': scores[4] + [scores[5]],
            'F1': scores[6] + [scores[7]]
    })

    results_table = pd.DataFrame(results).set_index('Label')

    with open(f'{tables_dir_name}/ml_results_table_{file_name}.tex','w') as tf:
        tf.write(results_table.to_latex())
    dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_{file_name}.png')

def create_and_save_aggregate_table(scores_lr, scores_rf, scores_svm, tables_dir_name, file_name):
    # Aggregated results
    results = ({
            'Metric' : ['Accuracy', 'Precision', 'Recall', 'F1'],
            'LR is_profitable': [mean(scores_lr[0]), mean(scores_lr[2]),  mean(scores_lr[4]), mean(scores_lr[6])],
            'RF is_profitable': [mean(scores_rf[0]), mean(scores_rf[2]),  mean(scores_rf[4]), mean(scores_rf[6])],
            'SVM is_profitable': [mean(scores_svm[0]), mean(scores_svm[2]),  mean(scores_svm[4]), mean(scores_svm[6])],
            'LR best_strategy': [scores_lr[1], scores_lr[3], scores_lr[5], scores_lr[7]],
            'RF best_strategy': [scores_rf[1], scores_rf[3], scores_rf[5], scores_rf[7]],
            'SVM best_strategy': [scores_svm[1], scores_svm[3], scores_svm[5], scores_svm[7]]
            })

    results_table = pd.DataFrame(results).set_index('Metric')

    with open(f'{tables_dir_name}/ml_results_table_{file_name}.tex','w') as f:
        f.write(results_table.to_latex())
    dfi.export(results_table.style, f'{tables_dir_name}/ml_results_table_{file_name}.png')

def create_and_save_feature_importance_plots(X_test, Y_test, labels, models, tables_dir_name, model_name, file_name):
    for i in range(0, len(labels)-1): # need to encode the last label and then add it back in
        result = permutation_importance(
            models[i], X_test, Y_test[labels[i]], n_repeats=10, random_state=50, n_jobs=2
        )
        importance_scores = result.importances_mean # use mean here ?
        importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance_scores})
        importance_df = importance_df.set_index('Feature').T

        exchange_cols = X_test.filter(regex=f'^exchange_', axis=1).columns
        state_cols = X_test.filter(regex=f'^state_', axis=1).columns
        country_cols = X_test.filter(regex=f'^country_', axis=1).columns
        sector_cols = X_test.filter(regex=f'^sector_', axis=1).columns
        industry_cols = X_test.filter(regex=f'^industry_', axis=1).columns

        importance_df['exchange'] = importance_df[X_test[exchange_cols].columns.tolist()].loc['Importance'].sum()
        importance_df['state'] = importance_df[X_test[state_cols].columns.tolist()].loc['Importance'].sum()
        importance_df['country'] = importance_df[X_test[country_cols].columns.tolist()].loc['Importance'].sum()
        importance_df['sector'] = importance_df[X_test[sector_cols].columns.tolist()].loc['Importance'].sum()
        importance_df['industry'] = importance_df[X_test[industry_cols].columns.tolist()].loc['Importance'].sum()

        importance_df = importance_df.drop(state_cols.tolist()+exchange_cols.tolist()+country_cols.tolist()+sector_cols.tolist()+industry_cols.tolist(), axis=1)
        importance_df = importance_df.T.reset_index()

        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.yticks(importance_df['Feature'], rotation=45, fontsize=5)
        plt.title(f'Feature Importance for label: {labels[i]} ({model_name})')
        plt.xlabel('Importances')
        plt.ylabel('Features')
        plt.savefig(f'{tables_dir_name}/feature_importance_{labels[i]}_{model_name}_{file_name}', bbox_inches='tight')
        plt.close()

def run_ml_models(training_data, testing_data, labels, tables_dir_name, file_name=''):
    imputer = SimpleImputer(strategy='mean')

    # Features
    X_train = imputer.fit_transform(training_data.drop(columns=labels))
    X_test = imputer.fit_transform(testing_data.drop(columns=labels))

    # Labels
    Y_train = training_data[labels].fillna(0)
    Y_test = testing_data[labels].fillna(0)

    regression_models, random_forest_models, svm_models = grid_search(X_train, Y_train, labels)
    save_model_params(labels, regression_models, random_forest_models, svm_models, tables_dir_name, file_name)

    # Feature importance
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(testing_data.drop(columns=labels)), columns=testing_data.drop(columns=labels).columns)
    create_and_save_feature_importance_plots(X, Y_test, labels, regression_models, tables_dir_name, 'regression', file_name)
    create_and_save_feature_importance_plots(X, Y_test, labels, random_forest_models, tables_dir_name, 'random_forest', file_name)
    create_and_save_feature_importance_plots(X, Y_test, labels, svm_models, tables_dir_name, 'svm', file_name)

    # Make predictions for each label on the test data
    y_pred_regression = make_predictions(X_test, regression_models)
    y_pred_random_forest = make_predictions(X_test, random_forest_models)
    y_pred_svm = make_predictions(X_test, svm_models)

    regression_scores = get_model_scores(Y_test, y_pred_regression)
    random_forest_scores = get_model_scores(Y_test, y_pred_random_forest)
    svm_scores = get_model_scores(Y_test, y_pred_svm)

    create_and_save_table(regression_scores, labels, tables_dir_name, f'regression_{file_name}')
    create_and_save_table(random_forest_scores, labels, tables_dir_name, f'random_forest_{file_name}')
    create_and_save_table(svm_scores, labels, tables_dir_name, f'svm_{file_name}')
    create_and_save_aggregate_table(regression_scores, random_forest_scores, svm_scores, tables_dir_name, f'aggregated_{file_name}')
