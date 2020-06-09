# written by kylesim
from argparse import ArgumentParser
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from util_text import *

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://scikit-learn.org/stable/modules/naive_bayes.html

# Apply Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable
# P(y|x1...xn) = P(x1...xn|y)P(y)/P(x1...xn)
# P(y|x1...xn) = P(x1|y)...p(xn|y)P(y)/P(x1...xn)
# P(y|x1...xn) = P(x1|y)...p(xn|y)P(y) => P(x1...xn) is constant
# y^ = argmax_y{P(x1|y)...P(xn|y)P(y)}

# Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. 
# It is also known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

# The Multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).

# The Complement Naive Bayes classifier was designed to correct the "severe assumptions" made by the standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets.

# GaussianNB implements the Gaussian Naive Bayes algorithm for classification.

# BernoulliNB implements the naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions; i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.

# CategoricalNB implements the categorical naive Bayes algorithm for categorically distributed data. It assumes that each feature, which is described by the index , has its own categorical distribution.


def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-c', '--classifier', type=str,
	default='mnb', help='( mnb | cnb | gnb)'
	)
	parser.add_argument(
	'-a', '--alpha', type=float,
	default=1.0, help='additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)'
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

	if PARAM.classifier == 'cnb':
		clf = ComplementNB()
	elif PARAM.classifier == 'mnb':
		clf = MultinomialNB()
	else:
		raise ValueError("[%s] not supported" % PARAM.classifier)

	clf_pipe = Pipeline([
		('tfidf', TfidfVectorizer()),
		('clf', clf)
	])

	gs_params = {
	'tfidf__max_features': [8000],
	'tfidf__min_df': [20],
	'tfidf__max_df': [0.7],
	'tfidf__ngram_range': [(1, 2)],
	'tfidf__analyzer': ['word'],
	'clf__alpha': [1.0, 0.5],
	}

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
	
	if PARAM.classifier == 'cnb':
		classifier = ComplementNB(alpha = PARAM.alpha)
	elif PARAM.classifier == 'gnb':
		classifier = GaussianNB()
	else:
		classifier = MultinomialNB(alpha = PARAM.alpha)
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

