import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv("movies.csv")

# convert genre text to numbers
encoder = LabelEncoder()
data["genre"] = encoder.fit_transform(data["genre"])

# select features
X = data[["rating","duration","popularity","genre"]]

# apply KMeans
kmeans = KMeans(n_clusters=3)
data["cluster"] = kmeans.fit_predict(X)

# show results
print(data[["movie","cluster"]])