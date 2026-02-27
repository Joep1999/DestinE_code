import sys
import optuna
import numpy as np
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")

from pathlib import Path

k_value = 9

BASE = Path("/srv/data/nas/project_data/p111_ecmwf_destine")
cluster_means = np.load(BASE / "machine_learning" / f"mean_clusters_k{k_value}.npy", allow_pickle=True)
cluster_dict = np.load(BASE / "machine_learning" / f"category_dict_k{k_value}.npy", allow_pickle=True)

labels = np.array(list(cluster_dict.item().values()))
keys = np.array(list(cluster_dict.item().keys()))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from verification_dashboard.data_access import load_saved_datasets, load_radar_avg
from verification_dashboard.metrics import domain_average
from datetime import datetime
import pandas as pd
import scoringrules as sr
# Make the prediction dataset
selected_keys = ['destine_ifs', 'nowcast']
timestep_interval = 30
timesteps = 24
#calculate CRPS over leadtimes ()

import scoringrules as sr

def calculate_crps_by_leadtime(radar_images, rainfall_images):
    
    crps_score = sr.crps_ensemble(radar_images, rainfall_images, m_axis = 0)
    crps_score_mean = np.nanmean(crps_score, axis = (0,1))
    
    return crps_score_mean


def build_training_data(date_str, multi_model = True):
    features = []
    date = datetime.strptime(date_str, "%Y%m%d%H")
    # --- Load NWP (no -1 hour shift) ---
    if multi_model:
        key_nwp = 'destine_ifs'
    else:
        key_nwp = 'destine'

    selected_keys = [key_nwp]
    series_mwp, metadata_radar_nwp = load_saved_datasets(
        selected_keys,
        date,
        timestep_interval,
        timesteps
    )
    # Use full forecast without excluding any timestep
    series_mwp_forecast = series_mwp[key_nwp]
    series_average = {}
    # --- Load nowcast (unchanged datetime, no slicing) ---
    selected_keys = ['nowcast']
    series_nowcast, metadata_radar_nowcast = load_saved_datasets(
        selected_keys,
        date,
        timestep_interval,
        timesteps
    )
    # Use full nowcast without excluding last timestep
    series_nowcast_forecast = series_nowcast['nowcast']
    # --- Domain averages ---
    series_average[key_nwp] = domain_average(series_mwp_forecast)
    series_average["nowcast"] = domain_average(series_nowcast_forecast)
    # --- Extract first timestep (shape: n_ensembles,) ---
    nwp_t0 = series_average[key_nwp][:, 0]      # 5 members
    nowcast_t0 = series_average["nowcast"][:, 0]      # 4 members
    # # 0 --- Ensemble mean difference (robust comparison) ---
    # mean_diff_t0 = nwp_t0.mean() - nowcast_t0.mean()
    # features.append(mean_diff_t0)
    # print(len(features))
    # # # --- Feature construction ---
    selected_keys = [key_nwp, 'nowcast']
    # # # 1. first 3 hours accumulation difference
    # features.append(series_average["destine_ifs"][:, :3].sum() - series_average["nowcast"][:, :3].sum())
    # print(len(features))
    # # 2. first timestep  difference
    # features.append(series_average["destine_ifs"][:, 0].mean() - series_average["nowcast"][:, 0].mean())
    # print(len(features))
    #3,4,5,6
    for key_variable in selected_keys:
        # Mean and standard deviation over entire series / ensembles
        features.append(series_average[key_variable].mean())
        #features.append(series_average[key_variable].std())
        mean_for_slope = series_average[key_variable].mean(axis=0)
        features.append(
            np.polyfit(range(len(mean_for_slope)), mean_for_slope, 1)[0]
        )
        print(len(features))
    
    radar_images = series_nowcast_forecast[0, 0]
    rainfall_images = series_mwp_forecast[:,0]
    features.append(calculate_crps_by_leadtime(radar_images, rainfall_images))
    return features
    
X_ml = []
y_ml = []

for key in keys:
    X_ml.append(build_training_data(key))
    y_ml.append(cluster_dict.item()[str(key)])

X_ml = np.array(X_ml)
y_ml = np.array(y_ml)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold

def run_sequential_feature_selection(X, y, n_features_to_select=5, forward=True):
    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # --- Base model ---
    clf = RandomForestClassifier(
        n_estimators=200,       # number of trees
        max_depth=3,            # shallow trees to prevent overfitting
        min_samples_leaf=3,     # ensures leaves have enough samples
        max_features='sqrt',    # reduce feature correlation
        random_state=0
    )
    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # --- Sequential Feature Selector ---
    sfs = SequentialFeatureSelector(
        clf,
        n_features_to_select=n_features_to_select,
        direction="forward" if forward else "backward",
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X_scaled, y)
    selected_mask = sfs.get_support()
    selected_indices = np.where(selected_mask)[0]
    print("Selected feature indices:", selected_indices)
    return selected_indices, scaler

all_selected = []
for i in range(3,10):
    selected_indices, scaler = run_sequential_feature_selection(
        X_ml, y_ml,
        n_features_to_select=i,   # choose how many you want
        forward=True
    )
    all_selected.append(selected_indices)

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np

loo = LeaveOneOut()

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=3,            # shallow trees to prevent overfitting
    min_samples_leaf=3,     # ensures leaves have enough samples
    max_features='sqrt',    # reduce feature correlation
    random_state=0
)

y_true = []
y_pred = []


##Sequential feature selection
# for list_features in all_selected:
#     y_true = []
#     y_pred = []
#     print(list_features)
#     for train_index, test_index in loo.split(X_ml):
#         X_train, X_test = X_ml[np.ix_(train_index, list_features)], X_ml[np.ix_(test_index, list_features)]
#         y_train, y_test = labels[train_index], labels[test_index]
#         clf.fit(X_train, y_train)
#         probs = clf.predict_proba(X_test)
#         # if probs[0][2] > 0.25:
#         #     predicted_class = [3]
#         # else:
#         predicted_class = np.argmax(probs, axis=1)
#         print('probs:', probs)
#         # print('predicted class:', predicted_class)
#         # print('actual class:',y_test[0])
#         y_true.append(y_test[0])
#         y_pred.append(predicted_class[0])
#     accuracy = balanced_accuracy_score(y_true, y_pred)
#     print("LOOCV Accuracy:", accuracy)

##################################



y_true = []
y_pred = []

pred_dict = {}

for train_index, test_index in loo.split(X_ml):
    X_train, X_test = X_ml[train_index], X_ml[test_index]
    
    y_train, y_test = labels[train_index], labels[test_index]
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    # if probs[0][2] > 0.25:
    #     predicted_class = [3]
    # else:
    predicted_class = np.argmax(probs, axis=1)
    # print('probs:', probs)
    # print('predicted class:', predicted_class)
    # print('actual class:',y_test[0])
    y_true.append(y_test[0])
    y_pred.append(predicted_class[0])

y_pred = list(y_pred)
keys = list(keys)

pred_dict = dict(zip(keys, y_pred))
np.save(BASE/ "machine_learning" / f"predictions_57_k{k_value}.npy", pred_dict, allow_pickle=True )

accuracy = balanced_accuracy_score(y_true, y_pred)
print("LOOCV Accuracy:", accuracy)


# train once on all data for operational use

import joblib

clf.fit(X_ml, y_ml)
# save the model to disk
filename = BASE / "machine_learning" / f"RF_model_57_k{k_value}.sav"
joblib.dump(clf, filename)


# re-test crps with these mean parameters to see how much worse they are compared to the optimal parameters. 
# If they are not much worse, then we can say that these clusters represent similar performance 
# and we can use the mean parameters for each cluster instead of the optimal parameters for each date.