# Song2Vec
A deep learning project that converts song embeddings into colored images, depending on the genres and tags of the song. 
With both embedding spaces we also trained a VAE model to compute the most optimal path between two songs.

# Getting Data

Go to the `data_scraping` dir and download `data.zip` and extract `songs.csv` and `tags.csv` into `./data/csv`

# Results

Results of our genre and tag embeddings:

### UMAP plots

![genre_embedding_space.png](outputs/plots/genre/genre_embedding_space.png)

![tag_embedding_space.png](outputs/plots/tags/tag_embedding_space.png)

### Cosine Similarity
![genre_cosine_similarity.png](outputs/plots/genre/genre_cosine_similarity.png)

![tag_cosine_similarity.png](outputs/plots/tags/tag_cosine_similarity.png)

### Nearest Neighbors check

![genre_nearest_neighbors.png](outputs/plots/genre/genre_nearest_neighbors.png)

![tag_nearest_neighbors.png](outputs/plots/tags/tag_nearest_neighbors.png)

# Visualizations

We made visualizations for these embeddings and tags together for songs. Here are some of them:

![multi_blobs.png](outputs/multi_blobs.png)

### VAE vs Linear comparison

![latent_path_Bossa_Antigua_to_Player_One.png](outputs/comparisons/latent_path_Bossa_Antigua_to_Player_One.png)

![morph_compare_Bossa_Antigua_to_Player_One.png](outputs/comparisons/morph_compare_Bossa_Antigua_to_Player_One.png)

![morph_compare_Bossa_Antigua_to_Player_One.gif](outputs/comparisons/morph_compare_Bossa_Antigua_to_Player_One.gif)
