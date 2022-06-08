import os
import sklearn.feature_extraction
import pandas as pd
import numpy as np
from tqdm import tqdm

import pe_features
my_extractor = pe_features.PEFileFeatures()

# Open a PE File and see what features we get
# filename = 'train/white/7032324E60BEC4DC4EBF7336D493E7E6'
# with open(filename,'rb') as f:
#     features = my_extractor.execute(f.read())
# features

# Load up all our files (files come from various places contagio, around the net...)
def load_files(file_list):
    features_list = []
    for filename in tqdm(file_list):
        with open(filename,'rb') as f:
            features_list.append(my_extractor.execute(f.read()))
    return features_list

#
file_list = [os.path.join('test', child) for child in tqdm(os.listdir('test'))]
test = load_files(file_list)
print 'Loaded up %d benign PE Files' % len(test)
import pandas as pd
test = pd.DataFrame.from_records(test)
test.head()


# Good (benign) files
file_list = [os.path.join('train/white', child) for child in tqdm(os.listdir('train/white'))]
good_features = load_files(file_list)
print 'Loaded up %d benign PE Files' % len(good_features)

# Bad (malicious) files
file_list = [os.path.join('train/black', child) for child in tqdm(os.listdir('train/black'))]
bad_features = load_files(file_list)
print 'Loaded up %d malicious PE Files' % len(bad_features)


# Putting the features into a pandas dataframe
import pandas as pd
df_bad = pd.DataFrame.from_records(bad_features)
df_bad['label'] = 'bad'
df_good = pd.DataFrame.from_records(good_features)
df_good['label'] = 'good'
df_good.head()

# Concatenate the info into a big pile!
df = pd.concat([df_bad, df_good], ignore_index=True)
df.replace(np.nan, 0, inplace=True)


# List of feature vectors (scikit learn uses 'X' for the matrix of feature vectors)
X = df.as_matrix(['number_of_import_symbols', 'number_of_sections'])

# Labels (scikit learn uses 'y' for classification labels)
y = np.array(df['label'].tolist())

import sklearn.ensemble
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=4)
print scores

my_seed = 123
my_tsize = .4 # 40%
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_tsize, random_state=my_seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
labels = ['good', 'bad']
cm = confusion_matrix(y_test, y_pred, labels)