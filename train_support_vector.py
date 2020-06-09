# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/svm.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# ----- guideline -----
# impractical for large datasets (slow)
# use LinearSVC, SGDClassifier for large datasets (fast)
# probability estimates not recommended

# The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using sklearn.svm.LinearSVC or sklearn.linear_model.SGDClassifier instead, possibly after a sklearn.kernel_approximation.Nystroem transformer.

# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--kernel', type=str,
	default='rbf', help='( rbf | poly | sigmoid | linear )'
	)
	parser.add_argument(
	'--gamma', type=str,
	default='scale', help='( scale | auto )'
	)
	parser.add_argument(
	'--df_shape', type=str,
	default='ovr', help='( ovo | ovr ) -> one-vs-one, one-vs-rest'
	)
	parser.add_argument(
	'-C', '--c_strength', type=float,
	default=1.0, help='inverse of regularization strength (smaller values specify stronger regularization)'
	)
	parser.add_argument(
	'--max_iter', type=int,
	default=1000, help='max number of iterations to be run'
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
	'--probability', action='store_true',
	default=False, help='enable probability estimates (slow)'
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
		('clf', SVC())
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	#'clf__max_iter': [500],
	'clf__C': [1.0, 10.0],
	'clf__kernel': ['rbf'],
	'clf__gamma': ['scale'],
	#'clf__gamma': ['scale', 'auto'],
	}

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("svc_%s" % (classifier.best_params_['clf__kernel'])))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "svc_%s" % (classifier.best_params_['clf__kernel'])
		model_map['accuracy'] = classifier.best_score_
		model_map['scaler'] = None
		model_map['vectorizer'] = None
		model_map['classifier'] = classifier
		save_model_map(model_map, PARAM.model_path)


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
	
	classifier = SVC(C = PARAM.c_strength, kernel = PARAM.kernel, gamma = PARAM.gamma, decision_function_shape = PARAM.df_shape, max_iter = PARAM.max_iter, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "svc_%s" % (PARAM.kernel)
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
