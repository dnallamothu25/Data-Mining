

import os
import nltk
nltk.download()
import ssl
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import sqrt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

corpusroot = './US_Inaugural_Addresses/'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

filename_tokens_dict = {}
raw_tf = {}
doc_tf = {}  # Dictionary to store TF values
doc_num = 0
term_df = {}  # Dictionary to store document frequency
tfidfnorm = {}
tokens_ns = {}

def getidf(token):
    stemmed_token = stemmer.stem(token.lower())
    if stemmed_token in term_df:
        df = len(term_df[stemmed_token])
        idf = math.log10(doc_num / df)
        return idf
    else:
        return -1

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        raw_tf_document = {}
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        doc = doc.lower()
        tokens = tokenizer.tokenize(doc)
        tokens_ns[filename] = tokens
        tokens_with_filename = [stemmer.stem(token) for token in tokens if token not in stop_words]
        filename_tokens_dict[filename] = tokens_with_filename

        # Calculate raw TF for each term in the document
        for term in tokens_with_filename:
            if term in raw_tf_document:
                raw_tf_document[term] += 1  # Increment the count for an existing term
            else:
                raw_tf_document[term] = 1  # Initialize the count for a new term

        # Calculate log TF for each term in the document
        log_tf = {term: 1 + math.log10(raw_tf_document[term]) for term in raw_tf_document}
        doc_tf[filename] = {'log_tf': log_tf}  # Store log TF values for this document
        doc_num += 1

for filename, tokens in filename_tokens_dict.items():
    for term in set(tokens):
        stemmed_token = stemmer.stem(term.lower())
        if stemmed_token not in term_df:
            term_df[stemmed_token] = set()  # Initialize as an empty set if not present
        term_df[stemmed_token].add(filename)

for term, filenames in term_df.items():
    term_df[term] = list(filenames)

term_idf = {}
for term, filenames in tokens_ns.items():
    idf_value = getidf(term)
    term_idf[term] = idf_value

for filename, doc_data in doc_tf.items():
    tfidfnorm[filename] = {}
    log_tf_dict = doc_data.get('log_tf', {})
    for term, tf in log_tf_dict.items():
        idf_value = getidf(term)

        if idf_value != -1:
            tfidf = tf * idf_value
            tfidfnorm[filename][term] = tfidf

def getweight(filename, token):
    token = token.lower()  # Lowercase the token first
    stemmed_token = stemmer.stem(token)  # Stem the token
    denum = 0  # Initialize the denominator for normalization
    if filename in tfidfnorm:
        if stemmed_token in tfidfnorm[filename]:
            tfidf = tfidfnorm[filename][stemmed_token]
            total_tfidf = sum(tfidfnorm[filename][term] ** 2 for term in tfidfnorm[filename])
            denum = sqrt(total_tfidf) if total_tfidf > 0 else 1.0  # Avoid division by zero
            normalized_tfidf = tfidf / denum if denum != 0 else 0.0
            return normalized_tfidf
    return 0.0

def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)
    norm_vector1 = math.sqrt(sum(val ** 2 for val in vector1.values()))
    norm_vector2 = math.sqrt(sum(val ** 2 for val in vector2.values()))

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0  # Avoid division by zero
    else:
        return dot_product / (norm_vector1 * norm_vector2)

def query(query_string):
    query_tokens = tokenizer.tokenize(query_string.lower())
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]
    query_tfidf = {}
    for term in query_tokens:
        raw_tf = query_tokens.count(term)
        log_tf = 1 + math.log10(raw_tf) if raw_tf > 0 else 0
        query_tfidf[term] = log_tf 

    euclidean_norm = math.sqrt(sum(tfidf ** 2 for tfidf in query_tfidf.values()))
    normalized_query_tfidf = {term: tfidf / euclidean_norm for term, tfidf in query_tfidf.items()}

    cosine_similarities = {}
    for filename in doc_tf:
        cosine_similarities[filename] = cosine_similarity(normalized_query_tfidf, tfidfnorm[filename])
    top_document = max(cosine_similarities, key=cosine_similarities.get)
    top_similarity = cosine_similarities[top_document]

    if top_similarity == 0:
        return "None", 0.0
    elif top_document == "None":
        return "None", 0.0
    else:
        return top_document, top_similarity

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))


# In[ ]:





# In[ ]:




