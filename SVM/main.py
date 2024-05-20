import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

df = pd.read_csv("creditcard.csv")

# Count the occurrences of fraud and no fraud and print them and their data visualization
occurrences = df['Class'].value_counts()
#Print the ratio of fraud cases
ratio_cases = occurrences/len(df.index)
#print(f'Ratio of fraudulent cases: {ratio_cases[1]}\nRatio of non-fraudulent cases: {ratio_cases[0]}')
#The ratio of fraudulent transactions is very low. This is a case of class imbalance problem

def prep_data(df: pd.DataFrame):
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y

X, y = prep_data(df)

scalar = StandardScaler()
scaled_df = scalar.fit_transform(X, y)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.metrics import zero_one_loss

from sklearn.svm import SVC


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0],
                    y=X[y == c1, 1],
                    alpha=.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=c1,
                    edgecolor='black'
                    )


np.random.seed(1)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)

svm = SVC(kernel='linear', random_state=1, gamma=1, C=1)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')

pred_y = svm.predict(X_xor)

error = zero_one_loss(y_xor, pred_y)

print("Zero one loss error:{}".format(error))

plt.show()
