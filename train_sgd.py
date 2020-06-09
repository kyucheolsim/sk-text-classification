# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/sgd.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ----- guideline -----
# 'hinge': linear SVM
# 'log': Logistic Regression
# 'modified_huber': smoothed hinge loss
# 'perceptron': perceptron

# Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data. For example, scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 and variance 1. Note that the same scaling must be applied to the test vector to obtain meaningful results. This can be easily done using StandardScaler:

# If your attributes have an intrinsic scale (e.g. word frequencies or indicator features) scaling is not needed.
# -> sometimes scaling does not improve performance

# Empirically, we found that SGD converges after observing approx. 10^6 training samples. Thus, a reasonable first guess for the number of iterations is max_iter = np.ceil(10**6 / n), where n is the size of the training set.

# Linear classifiers (SVM, logistic regression, a.o.) with SGD training.


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'-l', '--loss', type=str,
	default='log', help='( hinge | log | perceptron )'
	)
	parser.add_argument(
	'--penalty', type=str,
	default='l2', help='( l2 | l1 | elasticnet )'
	)
	parser.add_argument(
	'--n_gram', type=int,
	default=2, help='1 for (1, 1), 2 for (1, 2)'
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
	'--max_iter', type=int,
	default=70, help='max number of iterations'
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
		#('scaler', StandardScaler(with_mean=False)),
		#('scaler', QuantileTransformer(output_distribution='normal')),
		# -> sometimes scaling does not improve performance
		('clf', SGDClassifier(alpha = 0.0001, tol = 0.0001))
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	'clf__loss': ['hinge', 'log'],
	#'clf__penalty': ['l2', 'elasticnet'],
	'clf__max_iter': [70]
	}

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("sgd_%s" % (classifier.best_params_['clf__loss'])))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "sgd_%s" % (classifier.best_params_['clf__loss'])
		model_map['accuracy'] = classifier.best_score_
		model_map['scaler'] = None
		model_map['vectorizer'] = None
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


def get_n_gram_range(n):
	return (1, n)


def run_main():
	data = load_files(PARAM.data_path, encoding="utf-8")
	X, y, y_names = clean_docs(data.data, True), data.target, data.target_names

	vectorizer = tfidf_vectorize(X, max_features = PARAM.max_features, min_df = PARAM.min_df, max_df = PARAM.max_df, analyzer = PARAM.analyzer, ngram_range = get_n_gram_range(PARAM.n_gram))
	X_train, X_test, y_train, y_test = train_test_split(vectorizer.transform(X).toarray(), y, test_size = 0.2, random_state = 0)

	# use scaler for feature scaling
	scaler = get_scaler(PARAM.use_scaler, X_train)
	if scaler:
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

	if PARAM.loss == 'perceptron':
		# perceptron: perceptron
		# OPT => (eta0=1, learning_rate="constant", penalty=None)
		classifier = SGDClassifier(loss = 'perceptron', penalty = None, eta0 = 1.0, tol = 0.0001, n_iter_no_change = 5, max_iter = PARAM.max_iter, learning_rate = 'constant', early_stopping = False, random_state = 0)
	else:
		# hinge: linear SVM, log: Logistic Regression
		classifier = SGDClassifier(loss = PARAM.loss, penalty = PARAM.penalty, alpha = 0.0001, tol = 0.0001, n_iter_no_change = 5, max_iter = PARAM.max_iter, learning_rate = 'optimal', early_stopping = False, random_state = 0)

	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print("# iterations: {}".format(classifier.n_iter_))
	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "sgd_%s" (PARAM.loss)
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
