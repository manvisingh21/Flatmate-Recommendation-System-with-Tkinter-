import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv("C:/DOWNLOADS/FlatmateFinder.csv")  # Fix path separator

# Define features for content-based filtering
features = ['Age', 'Gender', 'Personality', 'Occupation', 'Tidiness Preference',
            'Dietary Preferences', 'Looking for (Gender)', 'Chore Preferences',
            'Personality Type', 'Lifestyle', 'Do you smoke?', 'Do you consume alcohol?',
            'Locality']

# Preprocess data
def preprocess_data(df):
    categorical_columns = features
    for column in categorical_columns:
        df[column] = df[column].str.lower()

preprocess_data(df.copy())  # Avoid modifying original data

# Apply TF-IDF vectorization (done outside functions for efficiency)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df[features])  # Combine features directly (optimized)

# Create KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Function to create dropdown menus using tkinter
def create_dropdown_menus(master, df):
    dropdown_menus = {}
    for i, feature in enumerate(features):
        label = ttk.Label(master, text=feature)
        label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
        
        values = sorted(df[feature].astype(str).unique())  # Get unique values as strings and sort them
        dropdown_menus[feature] = ttk.Combobox(master, values=values)
        dropdown_menus[feature].grid(row=i, column=1, padx=5, pady=5, sticky="ew")
        
        # Set default value for gender preference based on user's gender
        if feature == 'Gender':
            dropdown_menus[feature].set(values[0])  # Set to first value by default
        elif feature == 'Looking for (Gender)':
            user_gender = dropdown_menus['Gender'].get()
            dropdown_menus[feature] = ttk.Combobox(master, values=values)
            dropdown_menus[feature].grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            dropdown_menus[feature].set(user_gender)  # Set to user's gender by default
            
    return dropdown_menus

# Function to recommend flatmates based on user preferences
def recommend_flatmates(user_preferences, df, knn_model):
    recommended_flatmates_info = []

    # Combine user preferences into a single string
    user_features = ' '.join([str(user_preferences[feature]) for feature in features])

    # Apply TF-IDF vectorization to user features
    user_tfidf = tfidf_vectorizer.transform([user_features])

    # Find K nearest neighbors
    _, indices = knn_model.kneighbors(user_tfidf, n_neighbors=5+1)  # Adding 1 to get K+1 results, excluding the user itself
    recommended_indices = indices.flatten()[1:]  # Exclude the user itself

    # Filter recommended flatmates based on gender preference
    gender_preference = user_preferences.get('Looking for (Gender)')
    if gender_preference:
        gender_column = df['Gender']
        recommended_indices = [idx for idx in recommended_indices if gender_column.iloc[idx] == gender_preference]

    recommended_flatmates_info = [(df.iloc[idx], 100) for idx in recommended_indices]
    return recommended_flatmates_info

# Main program using tkinter
root = tk.Tk()
root.title("Flatmate Recommendation System")

# Create dropdown menus
dropdown_menus = create_dropdown_menus(root, df)

# Button to trigger recommendation on user input
def handle_button_click():
    user_preferences = {feature: dropdown_menus[feature].get() for feature in features}

    # Ensure selected gender preference is consistent across other preferences
    if 'Gender' in user_preferences:
        gender = user_preferences['Gender']
        for key in user_preferences.keys():
            if 'Looking for' in key and key != 'Looking for (Gender)':
                user_preferences[key] = gender

    # Generate recommendations
    recommended_flatmates = recommend_flatmates(user_preferences, df, knn_model)

    # Display recommended flatmates details (text-based output)
    print("Recommended Flatmates:")
    for flatmate, compatibility_percentage in recommended_flatmates:
        print(flatmate)
        print("Compatibility Percentage:", compatibility_percentage)
        print("\n")

recommend_button = ttk.Button(root, text="Recommend Flatmates", command=handle_button_click)
recommend_button.grid(row=len(features), columnspan=2, padx=10, pady=10)

root.mainloop()
