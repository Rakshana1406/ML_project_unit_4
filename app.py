import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- UI ---------------- #

st.set_page_config(page_title="Sports Team Analyzer", layout="wide")

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#ffb3ec,#c77dff,#a0c4ff);
}
h1,h2,h3{
color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 Smart Sports Team Performance Analyzer")

# ---------------- Load Data ---------------- #

data = pd.read_csv("teams.csv")

st.subheader("📊 Team Dataset")
st.dataframe(data)

# ---------------- Features ---------------- #

features = ["matches_won","matches_lost","points_scored",
            "defense_score","possession","rebounds","fouls"]

X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Clustering ---------------- #

model = AgglomerativeClustering(n_clusters=3)
data["cluster"] = model.fit_predict(X_scaled)

cluster_names = {
0:"🟢 Strong Team",
1:"🟡 Average Team",
2:"🔴 Weak Team"
}

data["cluster_name"] = data["cluster"].map(cluster_names)

# ---------------- Team Selection ---------------- #

st.subheader("🏀 Select Team")

team = st.selectbox("Choose Team", data["team"])

team_data = data[data["team"]==team]

st.subheader("📋 Team Details")

st.write(team_data)

# ---------------- Ranking ---------------- #

st.subheader("🏆 Team Ranking")

data["performance_score"] = (
data["matches_won"]*2 +
data["points_scored"]*0.01 +
data["defense_score"]*1.5 +
data["rebounds"]
)

ranking = data.sort_values(by="performance_score",ascending=False)

st.dataframe(ranking[["team","performance_score"]])

# ---------------- Team Comparison ---------------- #

st.subheader("⚔ Compare Teams")

team1 = st.selectbox("Team 1", data["team"])
team2 = st.selectbox("Team 2", data["team"], index=1)

t1 = data[data["team"]==team1]
t2 = data[data["team"]==team2]

compare = pd.concat([t1,t2])

st.dataframe(compare)

# ---------------- Graph ---------------- #

st.subheader("📊 Matches Won vs Points")

fig, ax = plt.subplots()

ax.scatter(
data["points_scored"],
data["matches_won"],
c=data["cluster"]
)

ax.set_xlabel("Points Scored")
ax.set_ylabel("Matches Won")

st.pyplot(fig)

# ---------------- PCA Visualization ---------------- #

st.subheader("📉 Cluster Visualization (PCA)")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

fig2, ax2 = plt.subplots()

ax2.scatter(
pca_data[:,0],
pca_data[:,1],
c=data["cluster"]
)

ax2.set_title("Team Clusters")

st.pyplot(fig2)