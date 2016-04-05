import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

def replace_non_numeric(clm):
    clm["v24"] = clm["v24"].apply(lambda v24: 0 if v24 == "A" else 1 if v24 == "B" else 2 if v24 == "C" else 3 if v24 == "D" else 4)
    clm["v30"] = clm["v30"].apply(lambda v30: 0 if v30 == "A" else 1 if v30 == "B" else 2 if v30 == "C" else 3 if v30 == "D" else 4 if v30 == "E" else 5 if v30 == "F" else 6)
    clm["v31"] = clm["v31"].apply(lambda v31: 0 if v31 == "A" else 1 if v31 == "B" else 2)
    clm["v47"] = clm["v47"].apply(lambda v47: 0 if v47 == "A" else 1 if v47 == "B" else 2 if v47 == "C" else 3 if v47 == "D" else 4 if v47 == "E" else 5 if v47 == "F" else 6 if v47 == "G" else 7 if v47 == "H" else 8 if v47 == "I" else 9)
    clm["v52"] = clm["v52"].apply(lambda v52: 0 if v52 == "A" else 1 if v52 == "B" else 2 if v52 == "C" else 3 if v52 == "D" else 4 if v52 == "E" else 5 if v52 == "F" else 6 if v52 == "G" else 7 if v52 == "H" else 8 if v52 == "I" else 9 if v52 == "J" else 10 if v52 == "K" else 11)
    clm["v66"] = clm["v66"].apply(lambda v66: 0 if v66 == "A" else 1 if v66 == "B" else 2)
    clm["v71"] = clm["v71"].apply(lambda v71: 0 if v71 == "B" else 1 if v71 == "C" else 2)
    clm["v74"] = clm["v74"].apply(lambda v74: 0 if v74 == "A" else 1 if v74 == "B" else 2)
    clm["v75"] = clm["v75"].apply(lambda v75: 0 if v75 == "A" else 1 if v75 == "B" else 2 if v75 == "C" else 3)
    clm["v79"] = clm["v79"].apply(lambda v79: 0 if v79 == "A" else 1 if v79 == "B" else 2 if v79 == "C" else 3 if v79 == "D" else 4 if v79 == "E" else 5 if v79 == "F" else 6 if v79 == "G" else 7 if v79 == "H" else 8 if v79 == "I" else 9 if v79 == "J" else 10 if v79 == "K" else 11 if v79 == "L" else 12 if v79 == "M" else 13 if v79 == "N" else 14 if v79 == "O" else 15 if v79 == "P" else 16 if v79 == "Q" else 17)
    clm["v91"] = clm["v91"].apply(lambda v91: 0 if v91 == "A" else 1 if v91 == "B" else 2 if v91 == "C" else 3 if v91 == "D" else 4 if v91 == "E" else 5 if v91 == "F" else 6)
    clm["v110"] = clm["v110"].apply(lambda v110: 0 if v110 == "A" else 1 if v110 == "B" else 2)
    return clm

train_df = replace_non_numeric(pd.read_csv("train.csv"))

train_df = pd.read_csv("train.csv")

et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

columns = ["v24","v30","v31","v47","v52","v66","v71","v74","v75","v79","v91","v110","v1", "v2", "v4", "v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v23","v25","v26","v27","v28","v29","v32","v33","v34","v35","v36","v37","v38","v39","v40","v41","v42","v43","v44","v45","v46","v48","v49","v50","v51","v53","v54","v55","v57","v58","v59","v60","v61","v62","v63","v64","v65","v67","v68","v69","v70","v72","v73","v76","v77","v78","v80","v81","v82","v83","v84","v85","v86","v87","v88","v89","v90","v92","v93","v94","v95","v96","v97","v98","v99","v100","v101","v102","v103","v104","v105","v106","v108","v109","v111","v114","v115","v116","v117","v118","v119","v120","v121","v122","v123","v124","v126","v127","v128","v129","v130","v131"]

labels = train_df["target"].values
features = train_df[list(columns)].values

et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()


print("{0} -> ET: {1})".format(columns, et_score))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)

test_df = replace_non_numeric(pd.read_csv("test.csv"))

et.fit(features, labels)

predictions = et.predict(imp.transform(test_df[columns].values))
test_df["PredictedProb"] = pd.Series(predictions)
test_df.to_csv("sample_submission.csv", cols=['ID', 'PredictedProb'], index=False)