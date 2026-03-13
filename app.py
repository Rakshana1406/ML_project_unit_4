import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- THEME ---------------- #

st.set_page_config(page_title="Tamil Movie Explorer", layout="wide")

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#ff9cee,#c77dff,#a0c4ff);
}

h1{
color:white;
text-align:center;
}

h2,h3{
color:white;
}

div.stButton > button{
background-color:#ff4dc4;
color:white;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("🎬 Smart Tamil Movie Explorer 🍿")
st.write("Discover movies using Machine Learning 🤖")

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv("movies.csv")

encoder = LabelEncoder()
data["genre_encoded"] = encoder.fit_transform(data["genre"])

X = data[["rating","duration","popularity","genre_encoded"]]

# ---------------- CLUSTERING ---------------- #

kmeans = KMeans(n_clusters=3, random_state=42)
data["cluster"] = kmeans.fit_predict(X)

# ---------------- MOVIE LIST ---------------- #

st.subheader("📜 Available Movies")

st.write("🍿 Choose your favorite movie below")

st.dataframe(data[["movie","genre","rating"]])

# ---------------- USER INPUT ---------------- #

movie = st.selectbox("🎥 Select a Movie", data["movie"])

mode = st.selectbox(
"⚙ Choose Recommendation Mode",
["⭐ Similar Movies","🎭 Same Genre","🔥 Popular Movies"]
)

# ---------------- MOVIE DETAILS ---------------- #

st.subheader("🎬 Movie Details")

details = data[data["movie"]==movie]

st.write("**🎥 Movie:**", details["movie"].values[0])
st.write("**🎭 Genre:**", details["genre"].values[0])
st.write("**⭐ Rating:**", details["rating"].values[0])
st.write("**⏱ Duration:**", details["duration"].values[0])
st.write("**🔥 Popularity:**", details["popularity"].values[0])

# ---------------- GRAPH ---------------- #

st.subheader("📊 Movie Cluster Graph")

fig, ax = plt.subplots()

scatter = ax.scatter(
    data["rating"],
    data["popularity"],
    c=data["cluster"]
)

ax.set_xlabel("⭐ Rating")
ax.set_ylabel("🔥 Popularity")
ax.set_title("Movie Clusters")

st.pyplot(fig)

# ---------------- RECOMMENDATION ---------------- #

st.subheader("🎯 Recommended Movies")

if mode == "⭐ Similar Movies":

    similarity = cosine_similarity(X)
    index = data[data["movie"]==movie].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores,key=lambda x:x[1],reverse=True)[1:4]

    for i in scores:
        st.write("🎬", data.iloc[i[0]]["movie"])

elif mode == "🎭 Same Genre":

    genre = details["genre"].values[0]
    result = data[data["genre"]==genre]["movie"]

    for m in result:
        st.write("🎬", m)

else:

    result = data.sort_values(by="popularity",ascending=False)["movie"].head(5)

    for m in result:
        st.write("🔥", m)

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.write("💡 Built using Machine Learning Clustering")