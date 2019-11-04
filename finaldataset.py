import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

#Read CSV File
filename_LCP = "dataset10.csv"
df_LCP = pd.read_csv(filename_LCP)

print(df_LCP.shape)
print(df_LCP.head())

#Data Frame Split x value, y value and train data:70% test data :30% classification

df = df_LCP.values
X = df[:,:6]
y = df[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#classifier Library define
dict_classifiers = {
    "Linear SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
}


# input : Train,Test X, y
# Traning Time set, Traning score, Test Score
def Score_classify(X_train, y_train, X_test, y_test, no_classifiers=5, verbose=True):

    Score_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, y_train)
        t_end = time.clock()

        t_diff = t_end - t_start
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        Score_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return Score_models

def display_Score_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)), columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    print(df_.sort_values(by=sort_by, ascending=False))


# Training Score, Test Score, Training Time Display
dict_models = Score_classify(X_train, y_train, X_test, y_test, no_classifiers=8)
display_Score_models(dict_models)

#improving upon the classifier
GDB_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.5, 0.1, 0.01, 0.001],
    'criterion': ['friedman_mse', 'mse', 'mae']
}
df = df_LCP.values
X = df[:,:6]
y = df[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#Correlation with lanemarker
def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0, len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()
display_corr_with_col(df_LCP, 'lanemarker')

ax = sns.pairplot(df_LCP, hue='lanemarker')
plt.title('Pairwise relationships between the features')
plt.show()


