# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# Multi-layer Perceptron classifier.
# This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. For example, scale each attribute on the input vector X to [0, 1] or [-1, +1], or standardize it to have mean 0 and variance 1. Note that you must apply the same scaling to the test set for meaningful results. You can use StandardScaler for standardization.

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--hidden_size', type=int,
	default=100, help='hidden layer size'
	)
	parser.add_argument(
	'--n_layers', type=int,
	default=2, help='number of hidden layers'
	)
	parser.add_argument(
	'--activation', type=str,
	default='relu', help='( logistic | tanh | relu )'
	)
	parser.add_argument(
	'--solver', type=str,
	default='adam', help='( lbfgs | sgd | adam )'
	)
	parser.add_argument(
	'--lr_schedule', type=str,
	default='constant', help='( constant | invscaling | adaptive )'
	)
	parser.add_argument(
	'--lr_init', type=float,
	default=0.001, help='initial learning rate'
	)
	parser.add_argument(
	'--max_iter', type=int,
	default=100, help='maximum number of iterations'
	)
	parser.add_argument(
	'--batch_size', type=int,
	default=100, help='size of minibatches for stochastic optimizers'
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
	'--use_scaler', type=int,
	default=0, help='( 0 | 1 | 2 | 3 ) -> 0: no scaler'
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
		('scaler', QuantileTransformer(output_distribution='normal')),
		('clf', MLPClassifier())
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	#'clf__hidden_layer_sizes': [(100,), (100, 100)],
	'clf__hidden_layer_sizes': [(100, 100)],
	'clf__max_iter': [100, 200],
	#'clf__learning_rate': ['invscaling', 'adaptive'],
	#'clf__activation': ['logistic', 'relu'],
	}

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("mlp"))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "mlp"
		model_map['accuracy'] = classifier.best_score_
		model_map['scaler'] = None
		model_map['vectorizer'] = None
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


def get_hidden_layer_size(n_layers, hidden_size):
	hls = tuple([hidden_size] * n_layers)
	return hls


def run_main():
	data = load_files(PARAM.data_path, encoding="utf-8")
	X, y, y_names = clean_docs(data.data, True), data.target, data.target_names
	vectorizer = tfidf_vectorize(X, max_features = PARAM.max_features, min_df = PARAM.min_df, max_df = PARAM.max_df, analyzer = PARAM.analyzer, ngram_range = (1, 2))
	X_train, X_test, y_train, y_test = train_test_split(vectorizer.transform(X).toarray(), y, test_size = 0.2, random_state = 0)

	# use scaler for feature scaling
	scaler = get_scaler(PARAM.use_scaler, X_train)
	if scaler:
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

	hls = get_hidden_layer_size(PARAM.n_layers, PARAM.hidden_size)
	classifier = MLPClassifier(hidden_layer_sizes = hls, early_stopping = False, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "mlp"
		model_map['accuracy'] = get_accuracy(y_test, y_pred)
		model_map['scaler'] = scaler
		model_map['vectorizer'] = vectorizer
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


if __name__ == '__main__':
	if PARAM.grid_search:
		run_grid_search()
	else:
		run_main()
