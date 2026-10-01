from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The government announced a new policy",
    "The government announced a new law",
    "Scientists discovered a new planet"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())