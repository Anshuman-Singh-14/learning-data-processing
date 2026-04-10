from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

docs = ["bad boring movie", "great good amazing", "waste of time", "loved it best"]
labels = [0, 1, 0, 1] # 0=Neg, 1=Pos

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
model = MultinomialNB().fit(X, labels)

pred = model.predict(X)
print(f"Accuracy: {accuracy_score(labels, pred)}")
print(f"Precision: {precision_score(labels, pred)}")
print(f"Recall: {recall_score(labels, pred)}")