# written by kylesim
from argparse import ArgumentParser
from sklearn.datasets import load_files
# for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
from sklearn.metrics import accuracy_score
from util_text import *

def get_param():
	parser = ArgumentParser()
	parser.add_argument(
	'-a', '--action', type=str,
	default='', help='( batch | accuracy )'
	)
	parser.add_argument(
	'-d', '--data_path', type=str,
	default='', help='input data path for prediction'
	)
	parser.add_argument(
	'-m', '--model_path', type=str,
	default='', help='path for trained model'
	)
	parser.add_argument(
	'--labeled', action='store_true',
	default=False, help='data labeled -> (x ... true_y)'
	)
	parser.add_argument(
	'--print_y', action='store_true',
	default=False, help='print input true labels (true_y) -> (x ... true_y pred_y)'
	)
	parser.add_argument(
	'--debug', action='store_true',
	default=False, help='print model map and input parameters'
	)
	return parser.parse_args()
PARAM = get_param()
if PARAM.debug:
	print("# "+str(PARAM))


def print_result(X, y, pred_labels):
	if y:
		# include true labels
		for i, label in enumerate(pred_labels):
			print("{}\t{}\t{}".format(X[i], y[i], label))
	else:
		for i, label in enumerate(pred_labels):
			print("{}\t{}".format(X[i], label))


def print_model_map(model_map):
	clf_name = model_map.get('clf_name', None)
	if clf_name:
		print("# Classifier Name: {}".format(clf_name))

	accuracy = model_map.get('accuracy', None)
	if accuracy:
		print("# Train Accuracy: {}".format(accuracy))


if __name__ == '__main__':
	if PARAM.model_path:
		model_map = load_model_map(PARAM.model_path)
		if PARAM.debug:
			print_model_map(model_map)
		scaler = model_map.get('scaler', None)
		vectorizer = model_map.get('vectorizer', None)
		classifier = model_map.get('classifier', None)
	else:
		raise ValueError('model_path not set')

	data = load_files(PARAM.data_path, encoding="utf-8")
	X_word, y, y_names = clean_docs(data.data, True), data.target, data.target_names

	if vectorizer:
		X = vectorizer.transform(X_word).toarray()
		if scaler:
			X = scaler.transform(X)
	else:
		# [vectorizer, scaler, classifier] in pipeline
		X = X_word
	pred_labels = classifier.predict(X)

	if PARAM.action == 'batch':
		if PARAM.print_y:
			print_result(X_word, y, pred_labels)
		else:
			print_result(X_word, None, pred_labels)
	elif PARAM.action == 'accuracy':
		accuracy = accuracy_score(y, pred_labels)
		print("# Accuracy: {}".format(accuracy))
	else:
		raise ValueError('unknown action')
