This Python script utilizes several libraries for implementing a flatmate recommendation system with Tkinter. 

- `tkinter` is employed for GUI development, providing dropdown menus and buttons for user interaction.
- `pandas` is utilized to load and preprocess the dataset, ensuring uniformity by converting text data to lowercase.
- `sklearn` is utilized for machine learning functionalities. Specifically, `TfidfVectorizer` transforms text data into numerical vectors using TF-IDF representation, facilitating compatibility calculation.
- `NearestNeighbors` from `sklearn.neighbors` is applied to construct a KNN model. This algorithm calculates the cosine similarity between feature vectors, allowing the identification of similar flatmates based on user preferences.
  
The KNN model utilizes a cosine distance metric to find the nearest neighbors to a given user profile. It operates by computing the cosine similarity between the user's preferences and the preferences of other flatmates. This similarity measure quantifies the cosine of the angle between two vectors, where a smaller angle denotes greater similarity. The algorithm returns the k nearest neighbors, excluding the user itself, and filters recommendations based on gender preference if specified. Finally, the system displays the recommended flatmates along with their compatibility percentages.
