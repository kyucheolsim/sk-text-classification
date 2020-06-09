# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# ----- guideline -----
# liblinear -> small datasets (fast, binary)
# sag -> large datasets (fast, multinomial)
# saga -> sparse & large datasets (multinomial)
# lbfgs -> small datasets (multinomial)

# Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# The solvers implemented in the class LogisticRegression are "liblinear", "newton-cg", "lbfgs", "sag" and "saga":

# The solver "liblinear" uses a coordinate descent (CD) algorithm, and relies on the excellent C++ LIBLINEAR library, which is shipped with scikit-learn. However, the CD algorithm implemented in liblinear cannot learn a true multinomial (multiclass) model; instead, the optimization problem is decomposed in a "one-vs-rest" fashion so separate binary classifiers are trained for all classes.

# The "lbfgs", "sag" and "newton-cg" solvers only support l2 regularization or no regularization, and are found to converge faster for some high-dimensional data. Setting multi_class to "multinomial" with these solvers learns a true multinomial logistic regression model, which means that its probability estimates should be better calibrated than the default "one-vs-rest" setting.

# The "sag" solver uses Stochastic Average Gradient descent. It is faster than other solvers for large datasets, when both the number of samples and the number of features are large.

# The "saga" solver is a variant of "sag" that also supports the non-smooth penalty="l1". This is therefore the solver of choice for sparse multinomial logistic regression. It is also the only solver that supports penalty="elasticnet".

# The "lbfgs" is an optimization algorithm that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm, which belongs to quasi-Newton methods. The "lbfgs" solver is recommended for use for small data-sets but for larger datasets its performance suffers.


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--solver', type=str,
	default='sag', help='( sag | saga | lbfgs | liblinear )'
	)
	parser.add_argument(
	'--multi_class', type=str,
	default='auto', help='( auto | ovr | multinomial )'
	)
	parser.add_argument(
	'--penalty', type=str,
	default='l2', help='( l1 | l2 | elasticnet )'
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
		('clf', LogisticRegression())
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	'clf__max_iter': [1000],
	'clf__solver': ['lbfgs'],
	'clf__C': [1.0, 10.0],
	#'clf__C': [1.0, 0.1, 10.0],
	#'clf__solver': ['sag', 'liblinear', 'lbfgs'],
	}

	classifier = GridSearchCV(clf_pipe, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("logreg_%s" % (classifier.best_params_['clf__solver'])))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "logreg_%s" % (classifier.best_params_['clf__solver'])
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

	classifier = LogisticRegression(C = PARAM.c_strength, solver = PARAM.solver, penalty = PARAM.penalty, multi_class = PARAM.multi_class, max_iter = PARAM.max_iter, tol = 0.0001, random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	print_eval(y_test, y_pred)
	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "logreg_%s" % (PARAM.solver)
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
