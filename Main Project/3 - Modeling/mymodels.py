from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def KNN1(X_train, y_train, X_test, y_test, N_JOBS=1):
	# Setup arrays to store train and test accuracies
	neighbors = np.arange(1, 10)
	train_accuracy = np.empty(len(neighbors))
	test_accuracy = np.empty(len(neighbors))

	# Loop over different values of k
	for i, k in enumerate(neighbors):
	    # Setup a k-NN Classifier with k neighbors: knn
	    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=N_JOBS)

	    # Fit the classifier to the training data
	    knn.fit(X_train,y_train)
	    
	    #Compute accuracy on the training set
	    train_accuracy[i] = knn.score(X_train, y_train)

	    #Compute accuracy on the testing set
	    test_accuracy[i] = knn.score(X_test, y_test)

	# Generate plot
	plt.title('k-NN: Varying Number of Neighbors')
	plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
	plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
	plt.legend()
	plt.xlabel('Number of Neighbors')
	plt.ylabel('Accuracy')
	plt.show()