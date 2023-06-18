import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def preprocess_data(df):
    df = df.drop(['DoctorID', 'AnaesthetistID'], axis=1)
    df = pd.get_dummies(df, columns=['Surgery Type', 'Anesthesia Type'])
    return df


def split_labels(df, label_column):
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    return X, y


def plot_preds_vs_real(y_test, y_pred, set='Test', savepath=None):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted {} set'.format(set))
    perfect_predictions = [y_test.min(), y_test.max()]
    plt.plot(perfect_predictions, perfect_predictions, color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"preds_vs_real_{set.lower()}.png"))
    plt.clf()


def evaluate_model(y_test, y_pred, set='Test', savepath=None):
    plot_preds_vs_real(y_test, y_pred, set, savepath=savepath)
    print(f'{set} set:')
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    r2 = r2_score(y_test, y_pred)
    print("R-squared Score:", r2)


def plot_feature_importances(importances, features, inds, savepath=None):
    plt.bar(range(len(importances)), importances[inds])
    plt.xticks(range(len(features)), features[inds], rotation=90)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "feature_importances.png"))
    plt.clf()


def plot_feature_target_correlations(feature_values, target, feature, savepath=None):
    plt.scatter(target, feature_values)
    plt.xlabel('Target')
    plt.ylabel(feature)
    plt.title(f'{feature} vs Target')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{feature}_vs_target.png"))
    plt.clf()


def importance_analysis(model, X, y, top_k=3, savepath=None):
    features = X.columns
    importances = model.feature_importances_
    inds = np.argsort(importances)[::-1]
    for i, index in enumerate(inds):
        print(f"{i + 1}. {X.columns[index]}: {importances[index]}")
    plot_feature_importances(importances, features, inds, savepath=savepath)
    for i in inds[:top_k]:
        feature = features[i]
        plot_feature_target_correlations(X[feature], y, feature, savepath=savepath)


if __name__ == '__main__':

    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the config
    config = load_config('config.json')

    # Load the data
    data = pd.read_csv(config['data_path'], index_col=0)

    # Preprocess the data
    data = preprocess_data(data)

    # Split label column from features
    X, y = split_labels(data, label_column='Duration in Minutes')

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=42)

    # Initialize a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=config["n_estimators"], random_state=42)

    # Fit the model on the training data
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    evaluate_model(y_train, y_train_pred, set='Train', savepath=results_dir)

    # Evaluate the model
    y_test_pred = rf.predict(X_test)
    evaluate_model(y_test, y_test_pred, set='Test', savepath=results_dir)

    # Get feature importances
    importance_analysis(rf, X, y, savepath=results_dir)

