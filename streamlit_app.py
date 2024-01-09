import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

recommender_pool_df = pd.read_csv('spotify_s3g3_rec_pool.csv')
recommender_pool_df = recommender_pool_df.sort_values('track_name')
recommender_pool_df['track_names_with_artist'] = recommender_pool_df['track_name'] + ' by ' + recommender_pool_df['artist_name']
recommender_pool_df = recommender_pool_df.drop_duplicates(subset=['track_names_with_artist'])
options = recommender_pool_df['track_names_with_artist'].to_list()
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','liveness', 'valence', 'tempo']

st.title("OPM Recommender Engine")
st.caption("by: Team Swifthrees")
st.image("https://img.freepik.com/free-vector/musical-pentagram-sound-waves-notes-background_1017-33911.jpg?w=1800&t=st=1698402872~exp=1698403472~hmac=5c67bc8bbf36f092b5ffb32a69f8ac13d10d9e76c53dc69b88f87b0aacb7cee5")
st.divider()

choice = st.selectbox(
    "Select Seed Track: ", options = options)

def get_cosine_dist(x,y):
    cosine_dist = 1 - cosine_similarity(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    return cosine_dist

if st.button("Submit", type="primary"):
    seed_track = recommender_pool_df.loc[recommender_pool_df['track_names_with_artist'] == choice].iloc[0]
    recommender_pool_df['cosine_similarity_features'] = recommender_pool_df.apply(lambda x: get_cosine_dist(x[feature_cols],seed_track[feature_cols]), axis=1)
    predicted_genre = seed_track['predicted_genre']
    recommended_tracks = recommender_pool_df[(recommender_pool_df['predicted_genre'] == predicted_genre) & (recommender_pool_df['track_names_with_artist'] != choice)].sort_values('cosine_similarity_features').reset_index()

    st.divider()
    st.markdown("<h5>Predicted Genre: </h5>", unsafe_allow_html = True)
    st.info(predicted_genre.upper())
    st.markdown("<br><h5>Similar Songs: </h5>", unsafe_allow_html = True)

    max_euclidean = recommended_tracks['cosine_similarity_features'].max() 
    min_euclidean = recommended_tracks['cosine_similarity_features'].min()
    recommended_tracks['cosine_similarity_features_scaled'] =  (1 - (recommended_tracks['cosine_similarity_features'] - min_euclidean) / (max_euclidean - min_euclidean)).abs()

    for i, track in recommended_tracks[:50].iterrows():
    	#st.info("\t🎵 " + track['track_names_with_artist'])
    	st.progress(value = track['cosine_similarity_features_scaled'], text= "\t🎵 " + str(i+1) + ". " + track['track_names_with_artist'])
