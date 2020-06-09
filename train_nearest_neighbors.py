# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/neighbors.html

# The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these.
# Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits and satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.

# Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

# The k-neighbors classification in KNeighborsClassifier is the most commonly used technique. The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.

# In cases where the data is not uniformly sampled, radius-based neighbors classification in RadiusNeighborsClassifier can be a better choice. The user specifies a fixed radius r, such that points in sparser neighborhoods use fewer nearest neighbors for the classification. For high-dimensional parameter spaces, this method becomes less effective due to the so-called "curse of dimensionality".

# weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point.


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-c', '--classifier', type=str,
	default='knn', help='( knn | rnn | nc)'
	)
	parser.add_argument(
	'-n', '--n_neighbors', type=int,
	default=7, help='number of neighbors to use for knn'
	)
	parser.add_argument(
	'-r', '--radius', type=float,
	default=1.0, help='range of parameter space for rnn'
	)
	parser.add_argument(
	'--weight_func', type=str,
	default='uniform', help='( uniform | distance )'
	)
	parser.add_argument(
	'--algorithm', type=str,
	default='auto', help='( auto | ball_tree | kd_tree | brute )'
	)
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--max_features', type=int,
	default=8000, help='consider only top max_features terms ordered by tf'
	)
	parser.add_argument(
	'--min_df', type=int,
	default=20, help='ignore terms that have a df lower than min_df (count)'
	)
	parser.add_argument(
	'--max_df', type=float,
	default=0.7, help='ignore terms that have a df higher than max_df (proportion)'
	)
	parser.add_argument(
	'--analyzer', type=str,
	default='word', help='( word | char | char_wb )'
	)
	parser.add_argument(
	'-m', '--model_path', type=str,
	default='', help='path to save trained model'
	)
	parser.add_argument(
	'--grid_search', action='store_true',
	default=False, help='grid search parameters'
	)
	parser.add_argument(
	'--debug', action='store_true',
	default=False, help='print input parameters'
	)
	return parser.parse_args()
PARAM = get_param()
if PARAM.debug:
	print("# "+str(PARAM))


def run_grid_search():
	data = load_files(PARAM.data_path, encoding="utf-8")
	X, y, y_names = clean_docs(data.data, True), data.target, data.target_names

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	#'clf__weights': ['uniform', 'distance'],
	#'clf__algorithm': ['auto', 'ball_tree'],
	}

	if PARAM.classifier == 'knn':
		clf = KNeighborsClassifier()
		gs_params['clf__n_neighbors'] = [7, 9]
	elif PARAM.classifier == 'rnn':
		clf = RadiusNeighborsClassifier()
		gs_params['clf__radius'] = [1.0, 2.0]
	else:
		raise ValueError("[%s] not supported" % PARAM.classifier)

	clf_pipe = Pipeline([
		('tfidf', TfidfVectorizer()),
		('clf', clf)
	])

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % (PARAM.classifier))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = PARAM.classifier
		model_map['accuracy'] = classifier.best_score_
		model_map['vectorizer'] = None
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


def run_main():
	data = load_files(PARAM.data_path, encoding="utf-8")
	X, y, y_names = clean_docs(data.data, True), data.target, data.target_names
	vectorizer = tfidf_vectorize(X, max_features = PARAM.max_features, min_df = PARAM.min_df, max_df = PARAM.max_df, analyzer = PARAM.analyzer, ngram_range = (1, 2))
	X_train, X_test, y_train, y_test = train_test_split(vectorizer.transform(X).toarray(), y, test_size = 0.2, random_state = 0)
	
	if PARAM.classifier == 'knn':
		classifier = KNeighborsClassifier(n_neighbors = PARAM.n_neighbors, weights = PARAM.weight_func)
	elif PARAM.classifier == 'rnn':
		classifier = RadiusNeighborsClassifier(radius = PARAM.radius)
	elif PARAM.classifier == 'nc':
		classifier = NearestCentroid()
	else:
		raise ValueError("[%s] not supported" % (PARAM.classifier))

	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = PARAM.classifier
		model_map['accuracy'] = get_accuracy(y_test, y_pred)
		model_map['vectorizer'] = vectorizer
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


if __name__ == '__main__':
	if PARAM.grid_search:
		run_grid_search()
	else:
		run_main()

