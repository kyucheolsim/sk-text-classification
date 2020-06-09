# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
#from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='./data/txt_sentoken', help='input data path'
	)
	parser.add_argument(
	'--voting', type=str,
	default='soft', help='( hard | soft )'
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
	X_word, y, y_names = clean_docs(data.data, True), data.target, data.target_names
	vectorizer = tfidf_vectorize(X_word, max_features = PARAM.max_features, min_df = PARAM.min_df, max_df = PARAM.max_df, analyzer = PARAM.analyzer, ngram_range = (1, 2))
	X = vectorizer.transform(X_word).toarray()

	# use scaler for feature scaling
	scaler = get_scaler(PARAM.use_scaler, X)
	if scaler:
		X = scaler.transform(X)

	# three classifiers and voting classifier
	lr_clf = LogisticRegression(solver = "sag", multi_class = 'auto')
	rf_clf = RandomForestClassifier()
	svc_clf = SVC(kernel = "rbf", gamma = 'scale')
	vt_clf = VotingClassifier(
			estimators = [('lr', lr_clf), ('rf', rf_clf), ('svc', svc_clf)],
			voting = PARAM.voting, weights=[1,1,1])

	gs_params = {
	#'lg__C': [0.1, 1.0],
	#'lg__solver': ['sag', 'lbfgs'],
	'rf__n_estimators': [50, 100],
	}

	classifier = GridSearchCV(vt_clf, cv = 5, param_grid = gs_params, verbose = 1)
	classifier = classifier.fit(X, y)

	print("")
	print("# Classifier: %s" % ("voting_%s" % PARAM.voting))
	print("best score: %.3f\n" % (classifier.best_score_))
	print("# Best Parameters")
	for param in sorted(gs_params.keys()):
		print("%s: %r" % (param, classifier.best_params_[param]))
	print("")

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "voting_%s" % (PARAM.voting)
		model_map['accuracy'] = classifier.best_score_
		model_map['scaler'] = scaler
		model_map['vectorizer'] = vectorizer
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

	# three classifiers and voting classifier
	clf_01 = LogisticRegression(C = PARAM.c_strength, solver = "sag", multi_class = 'auto')
	clf_02 = RandomForestClassifier(n_estimators = 100)
	clf_03 = SVC(C = PARAM.c_strength, kernel = "rbf", gamma = 'scale')
	classifier = VotingClassifier(
			estimators = [('clf01', clf_01), ('clf02', clf_02), ('clf03', clf_03)], voting = PARAM.voting, weights=[1,1,1])

	# classifier 01
	clf_01.fit(X_train, y_train)
	y_pred = clf_01.predict(X_test)
	print_eval(y_test, y_pred)
	print_line()

	# classifier 02
	clf_02.fit(X_train, y_train)
	y_pred = clf_02.predict(X_test)
	print_eval(y_test, y_pred)
	print_line()

	# classifier 03
	clf_03.fit(X_train, y_train)
	y_pred = clf_03.predict(X_test)
	print_eval(y_test, y_pred)
	print_line()
	
	# voting classifier
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	print_eval(y_test, y_pred)

	if PARAM.model_path:
		model_map = {}
		model_map['clf_name'] = "voting_%s" % (PARAM.voting)
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
