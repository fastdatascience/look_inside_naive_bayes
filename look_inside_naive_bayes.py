from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.datasets import fetch_20newsgroups

vectoriser = CountVectorizer(lowercase=True, 
                         max_features=1000)
transformer = TfidfTransformer()

nb = MultinomialNB()
model = make_pipeline(vectoriser, transformer, nb)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

model.fit( twenty_train['data'],  twenty_train['target'])

def look_inside_naive_bayes(model):
    vectoriser= model.named_steps['countvectorizer']
    transformer= model.named_steps['tfidftransformer']
    nb= model.named_steps['multinomialnb']

    fake_document = " ".join(vectoriser.vocabulary_)
    vectorised_document = vectoriser.transform([fake_document])
    transformed_document = transformer.transform(vectorised_document)

    probas = np.zeros((transformed_document.shape[1]))

    vocab_idx_to_string_lookup = [""] * transformed_document.shape[1]
    for w, i in vectoriser.vocabulary_.items():
        vocab_idx_to_string_lookup[i] = w

    transformed_documents = np.zeros((transformed_document.shape[1], transformed_document.shape[1]))
    for i in range(transformed_document.shape[1]):
        transformed_documents[i, i] = transformed_document[0, i]

    probas_for_vocab_and_class = nb.predict_log_proba(transformed_documents)


    for prediction_idx, label in enumerate(model.classes_):
        print(f"Strongest predictors for class {label}\n")
        probas_this_class = probas_for_vocab_and_class[:, prediction_idx]

        top_vocab_idxes_this_class = np.argsort(-probas_this_class)

        for ctr, j in enumerate(top_vocab_idxes_this_class[:10]):
            word = vocab_idx_to_string_lookup[j]
            print(f"{ctr}\t{word}")

look_inside_naive_bayes(model)
