import sys
import optuna
sys.path.insert(0, "/home/joep/git/wi-research/p111_ecmwf_destine/verification_dashboard")

from data_access import load_settings_weights_blending
import os

optimal_pars_path = '/srv/data/nas/project_data/p111_ecmwf_destine/optimal_parameters/'
storage = f"sqlite:///optuna_blending_study.db"
study_names = optuna.study.get_all_study_names(storage=storage)

optimal_weights_dict = {}
for study in study_names:
    weights = load_settings_weights_blending(study, storage)
    study_date = study[-10:]
    optimal_weights_dict[study_date] = weights

    

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
        [entry['use_noise']],
       [entry['use_probmatching']]
    ])
    print(output)
    return output


import numpy as np

X = []
keys = []

for k, v in optimal_weights_dict.items():
    print(k,v)
    X.append(extract_features(v))
    keys.append(k)

X = np.array(X)

result = extract_features(optimal_weights_dict['2024080105'])

from sklearn.cluster import KMeans

k = 8
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X)

category_dict = dict(zip(keys, labels))

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


for i in range(k):
    mean_feat = X[labels == i].mean(axis=0)
    mean_noise = round(mean_feat[-2])  # round use_noise to nearest integer
    mean_probmatch = round(mean_feat[-1])  # round use_probmatching
    mean_gamma = mean_feat[:4].reshape(2,2)
    mean_regr  = mean_feat[4:-2].reshape(2,2)

    print(f"\nCluster {i}")
    print("Mean GAMMA:\n", mean_gamma)
    print("Mean regr_pars:\n", mean_regr)
    print("Mean use_noise:\n", mean_noise)
    print("Mean use_probmatching:\n", mean_probmatch)

# re-test crps with these mean parameters to see how much worse they are compared to the optimal parameters. 
# If they are not much worse, then we can say that these clusters represent similar performance 
# and we can use the mean parameters for each cluster instead of the optimal parameters for each date.