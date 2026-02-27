import sys
import optuna
import numpy as np
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/")
from verification_dashboard.data_access import load_settings_weights_blending
import os

data_path = '/srv/data/nas/project_data/p111_ecmwf_destine/'
optimal_pars_path = '/srv/data/nas/project_data/p111_ecmwf_destine/optimal_parameters/'
storage = f"sqlite:///optuna_blending_study.db"
study_names = optuna.study.get_all_study_names(storage=storage)

optimal_weights_dict = {}
for study in study_names:
    weights = load_settings_weights_blending(study, storage)
    study_date = study[-10:]
    optimal_weights_dict[study_date] = weights


#extract features from weights
def extract_features_no_regr_pars(entry):
    if len(entry["GAMMA"].flatten()) != 4 or len(entry["regr_pars"].flatten()) != 4:
        print('taking only the needed pars')
        G = entry["GAMMA"][:-2]          # (2,2)
        #R = entry["regr_pars"][:, :-2]   # (2,2)
    else:
        print('taking all pars')
        G = entry["GAMMA"]                # (2,2)
        #R = entry["regr_pars"]            # (2,2)
    output = np.concatenate([
        G.flatten(), 
        #R.flatten(), 
        #[entry['use_noise']],
       [entry['use_probmatching']]
    ])
    print(output)
    return output

#extract features from weights
def extract_features(entry):
    if len(entry["GAMMA"].flatten()) != 4 or len(entry["regr_pars"].flatten()) != 4:
        print('taking only the needed pars')
        G = entry["GAMMA"][:-2]          # (2,2)
        R = entry["regr_pars"][:, :-2]   # (2,2)
    else:
        print('taking all pars')
        G = entry["GAMMA"]                # (2,2)
        R = entry["regr_pars"]            # (2,2)
    output = np.concatenate([
        G.flatten(), 
        R.flatten(), 
        #[entry['use_noise']],
       [entry['use_probmatching']]
    ])
    print(output)
    return output


import numpy as np

X_no_regr_pars = []
keys = []

for k, v in optimal_weights_dict.items():
    print(k,v)
    X_no_regr_pars.append(extract_features_no_regr_pars(v))
    keys.append(k)

X_no_regr_pars = np.array(X_no_regr_pars)


X = []
keys = []

for k, v in optimal_weights_dict.items():
    print(k,v)
    X.append(extract_features(v))
    keys.append(k)

X = np.array(X)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_regr_pars)


from sklearn.cluster import KMeans

########## for k9################
k = 9
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X_scaled)

import joblib

joblib.dump(kmeans, data_path + "machine_learning/kmeans_model_k9.pkl")
joblib.dump(scaler, data_path + "machine_learning/scaler_k9.pkl")

category_dict = dict(zip(keys, labels))
np.save(data_path + "machine_learning/category_dict_k9.npy", category_dict)

mean_clusters = {}
for i in range(k):
    mean_feat = X[labels == i].mean(axis=0)
    #mean_noise = round(mean_feat[-2])  # round use_noise to nearest integer
    probmatch = round(mean_feat[-1])
    if probmatch == 1:
        mean_probmatch = True
    elif probmatch == 0:
        mean_probmatch = False
    mean_gamma = mean_feat[:4].reshape(2,2)
    mean_regr  = mean_feat[4:-1].reshape(2,2)
    mean_clusters[i] = {
        "GAMMA": mean_gamma,
        "regr_pars": mean_regr,
        "use_probmatching": mean_probmatch
    }
    print(f"\nCluster {i}")
    print("Mean GAMMA:\n", mean_gamma)
    print("Mean regr_pars:\n", mean_regr)
    #print("Mean use_noise:\n", mean_noise)
    print("Mean use_probmatching:\n", mean_probmatch)

np.save(data_path + "machine_learning/mean_clusters_k9.npy", mean_clusters)

########## for k3################
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, data_path + "machine_learning/kmeans_model_k3.pkl")
joblib.dump(scaler, data_path + "machine_learning/scaler_k3.pkl")

category_dict = dict(zip(keys, labels))
np.save(data_path + "machine_learning/category_dict_k3.npy", category_dict)



mean_clusters = {}
for i in range(k):
    mean_feat = X[labels == i].mean(axis=0)
    #mean_noise = round(mean_feat[-2])  # round use_noise to nearest integer
    probmatch = round(mean_feat[-1])
    if probmatch == 1:
        mean_probmatch = True
    elif probmatch == 0:
        mean_probmatch = False

    mean_gamma = mean_feat[:4].reshape(2,2)
    mean_regr  = mean_feat[4:-1].reshape(2,2)
    mean_clusters[i] = {
        "GAMMA": mean_gamma,
        "regr_pars": mean_regr,
        "use_probmatching": mean_probmatch
    }

    print(f"\nCluster {i}")
    print("Mean GAMMA:\n", mean_gamma)
    print("Mean regr_pars:\n", mean_regr)
    #print("Mean use_noise:\n", mean_noise)
    print("Mean use_probmatching:\n", mean_probmatch)

np.save(data_path + "machine_learning/mean_clusters_k3.npy", mean_clusters)




##############################
#Make plots to bverify the clustering results.
################################


import matplotlib.pyplot as plt

inertia = []
Ks = range(2,15)

for k in Ks:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(Ks, inertia, "-o")
plt.xlabel("Number of clusters")
plt.ylabel("Within-cluster variance")
plt.savefig(
                f"/srv/data/nas/project_data/p111_ecmwf_destine/verification/k_means_cluster.png",
                dpi=300,
                bbox_inches="tight",
            )
k=9
for i in range(k):
    mean_feat = X[labels == i].mean(axis=0)
    #mean_noise = round(mean_feat[-2])  # round use_noise to nearest integer
    mean_probmatch = round(mean_feat[-1])  # round use_probmatching
    mean_gamma = mean_feat[:4].reshape(2,2)
    mean_regr  = mean_feat[4:-1].reshape(2,2)

    print(f"\nCluster {i}")
    print("Mean GAMMA:\n", mean_gamma)
    print("Mean regr_pars:\n", mean_regr)
    #print("Mean use_noise:\n", mean_noise)
    print("Mean use_probmatching:\n", mean_probmatch)