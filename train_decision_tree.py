# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/tree.html

# Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
# Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
# Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

# DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.
# CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.
# scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--splitter', type=str,
	default='best', help='( best | random )'
	)
	parser.add_argument(
	'--criterion', type=str,
	default='gini', help='( gini | entropy ) -> Gini impurity or information gain'
	)
	parser.add_argument(
	'--max_depth', type=int,
	default=5, help='max depth of the tree'
	)
	parser.add_argument(
	'--max_split_features', type=str,
	default='auto', help='( auto | sqrt | log2 )'
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

	clf_pipe = Pipeline([
		('tfidf', TfidfVectorizer()),
		('clf', DecisionTreeClassifier())
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	#'clf__criterion': ['gini', 'entropy'],
	#'clf__splitter': ['best', 'random'],
	#'clf__max_features': ['auto', 'log2', None],
	#'clf__max_depth': [5, 7]
	#'clf__min_samples_split': [2, 5],
	'clf__min_samples_leaf': [1, 3],
	}

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("decision_tree"))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "decision_tree"
		model_map['accuracy'] = classifier.best_score_
		model_map['vectorizer'] = None
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


def run_main():
	data = load_files(PARAM.data_path, encoding="utf-8")
	X, y, y_names = clean_docs(data.data, True), data.target, data.target_names
	vectorizer = tfidf_vectorize(X, max_features = PARAM.max_features, min_df = PARAM.min_df, max_df = PARAM.max_df, analyzer = PARAM.analyzer, ngram_range = (1, 2))
	X_train, X_test, y_train, y_test = train_test_split(vectorizer.transform(X).toarray(), y, test_size = 0.2, random_state = 0)
	
	classifier = DecisionTreeClassifier(max_depth = PARAM.max_depth, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "decision_tree"
		model_map['accuracy'] = get_accuracy(y_test, y_pred)
		model_map['vectorizer'] = vectorizer
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


if __name__ == '__main__':
	if PARAM.grid_search:
		run_grid_search()
	else:
		run_main()
