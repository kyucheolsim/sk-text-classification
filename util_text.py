# written by kylesim
import re
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# sample dataset
# http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
# http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

RM_SYMBOLS = r'[~!@#$%^&*()\-_=+\[{\]}\\|;:\'\",<.>/?`]'

def clean_docs(X, mask_numbers = False):
	docs = []
	for x in X:
		doc = clean_string(x, mask_numbers)
		docs.append(doc)
	return docs


def clean_string(string, mask_numbers = False):
	# remove all the special characters
	# SYMBOLS_32 => ~!@#$%^&*()-_=+[{]}\|;:'",<.>/?`
	#string = re.sub(r'\W', ' ', string) # _ not removed
	string = re.sub(RM_SYMBOLS, ' ', string)
	
	# substitute multiple spaces with a single space
	string = re.sub(r'\s+', ' ', string)

	if mask_numbers:
		string = re.sub(r'\d+', '00', string)

	# convert to lowercase letters
	string = string.lower()

	# remove all leading and trailing whitespaces
	return string.strip()


def count_vectorize(docs, max_features = 30000, min_df = 1, max_df = 1.0, analyzer = 'word', ngram_range = (1, 1)):
	# bag of words model (word -> number)
	vectorizer = CountVectorizer(max_features = max_features, min_df = min_df, max_df = max_df, analyzer = analyzer, ngram_range = ngram_range)
	vectorizer = vectorizer.fit(docs)
	#numeric_docs = vectorizer.fit_transform(docs).toarray()
	#feature_names = vectorizer.get_feature_names()
	#stop_words = vectorizer.get_stop_words()
	#vocab = vectorizer.vocabulary_
	return vectorizer


def tfidf_vectorize(docs, max_features = 30000, min_df = 1, max_df = 1.0, analyzer = 'word', ngram_range = (1, 1)):
	# tfidf model (word -> number)
	# TF(word) = (Number of Occurrences of a word)/(Total words in the document)
	# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))
	# OPT: smooth_idf = True (Default)
	# => idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1
	# OPT: smooth_idf = False
	# => idf(t) = log [ n / df(t) ] + 1
	# OPT: sublinear_tf = True
	# => tf = 1 + log(tf)
	vectorizer = TfidfVectorizer(max_features = max_features, min_df = min_df, max_df = max_df, analyzer = analyzer, ngram_range = ngram_range)
	vectorizer = vectorizer.fit(docs)
	#numeric_docs = vectorizer.fit_transform(docs).toarray()
	#feature_names = vectorizer.get_feature_names()
	#stop_words = vectorizer.get_stop_words()
	#vocab = vectorizer.vocabulary_
	#idf = vectorizer.idf_
	return vectorizer


def get_scaler(use_scaler, X):
	if use_scaler == 1:
		scaler = StandardScaler().fit(X)
		# z = (x - u) / s
	elif use_scaler == 2:
		scaler = RobustScaler().fit(X)
		# robust to outliers (mean -> median,  sd -> IQR)
	elif use_scaler == 3:
		scaler = QuantileTransformer(output_distribution='normal').fit(X)
		# map data to normal distribution
	else:
		scaler = None
	return scaler


def get_accuracy(y_test, y_pred):
	return accuracy_score(y_test, y_pred)


def print_eval(y_test, y_pred):
	print("# Confusion Matrix")
	# true_neg (0,0), false_pos (0,1)
	# false_neg (1,0), true_pos (1,1)
	conf_mat = confusion_matrix(y_test, y_pred)
	#t_neg, f_pos, f_neg, t_pos = conf_mat.ravel()
	#print((t_neg, f_pos, f_neg, t_pos))
	print(conf_mat)
	print("")

	print("# Precision & Recall & F1")
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	print("precision: {:.3f}".format(precision))
	print("recall: {:.3f}".format(recall))
	print("f1-score: {:.3f}".format(f1_score(y_test, y_pred)))
	#print("f1-score: {:.3f}".format((2.0 * precision * recall)/(precision + recall)))
	print("")

	print("# Classification Report")
	print(classification_report(y_test, y_pred, digits = 3))

	print("# Accuracy")
	print(accuracy_score(y_test, y_pred))


def save_model_map(model, model_path):
	with open(model_path, 'wb') as model_file:
		pickle.dump(model, model_file)


def load_model_map(model_path):
	with open(model_path, 'rb') as model_file:
		return pickle.load(model_file)


def print_line(line_char = '-', how_many = 80):
	print(line_char*how_many)


if __name__ == "__main__":
	import pprint
	print_line()
	#noisy_line = "it is a noisy string ~!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`"
	docs = [
	"This is the first document.",
	"This is the second document.",
	"This is the 2nd document?",
	"This is the third document.",
	"This is the 3rd document?",
	"What is the 3rd document?",
	]

	test_docs = [
	"is this the first document?",
	"is this the second document?",
	"is this the 3rd document?",
	]

	pprint.pprint(docs)
	print_line()

	vector = tfidf_vectorize(docs, analyzer = 'word', ngram_range = (1, 2))
	feature_names = vector.get_feature_names()
	print(feature_names)
	print_line()
	pprint.pprint(vector.transform(docs).toarray())

	print_line()
	pprint.pprint(test_docs)
	print_line()
	print(feature_names)
	print_line()
	pprint.pprint(vector.transform(test_docs).toarray())

