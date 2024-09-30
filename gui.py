import pandas as pd
import tkinter as tk
from tkinter import ttk
import pickle


# Load the dataset
df = pd.read_csv("C:\\Users\\91939\\Downloads\\Mini Project - all files\\final mini\\binary_with_artist.csv", low_memory=False)

# Remove unnecessary spaces in column names
df.columns = df.columns.str.strip()

# Extract necessary data
songs = df['name'].tolist()  # List of available songs
song_info = df[['name', 'mood', 'artist', 'year']]

# Load the pre-trained Logistic Regression model from the .pkl file
with open("C:\\Users\\91939\\Downloads\\Mini Project - all files\\final mini\\logistic_model.pkl", 'rb') as file:
    logistic_model = pickle.load(file)

# Define mood colors based on available moods
mood_colors = {
    'devotional': '#FFD700',  # Gold for devotional songs
    'sad': '#ADD8E6',  # Light blue for sad songs
    'romantic': '#FFC0CB',  # Pink for romantic songs
    'upbeat': '#FFCC99'  # Light orange for upbeat songs
}


# Function to find the song's mood using the logistic model and display the details
def find_mood():
    selected_song = song_var.get()

    # Get the corresponding row from the dataset for the selected song
    song_details = song_info[song_info['name'] == selected_song]

    if not song_details.empty:
        # Get the artist and year
        artist = song_details['artist'].values[0]
        year = song_details['year'].values[0]

        # Extract features for prediction (use the same columns as when training)
        song_features = df[df['name'] == selected_song].drop(['index', 'name', 'mood', 'year', 'artist', 'lyrics'],
                                                             axis=1)

        # Predict the mood using the logistic regression model
        predicted_mood = logistic_model.predict(song_features)[0]

        # Update label with the song details
        result_label.config(text=f"Mood: {predicted_mood}\nArtist: {artist}\nYear: {year}")

        # Change background color based on predicted mood
        mood_color = mood_colors.get(predicted_mood.lower(), '#FFFFFF')  # Default to white if mood not found
        result_label.config(bg=mood_color)
    else:
        result_label.config(text="Song not found", bg='white')


# Create the main window
root = tk.Tk()
root.title("Song Mood Finder")
root.geometry("500x400")

# Set background color for the window
root.configure(bg="#F0F0F0")

# Song dropdown menu
song_var = tk.StringVar(root)
song_var.set(songs[0])  # Set default value

title_label = tk.Label(root, text="TeSonance: Mood Analysis of Telugu Songs", font=("Arial", 20, "bold"),
                       bg="#F0F0F0")
title_label.pack(pady=20)
# Label for dropdown
label = tk.Label(root, text="Select a song:", font=("Arial", 14), bg="#F0F0F0")
label.pack(pady=10)

# Dropdown box
dropdown = ttk.Combobox(root, textvariable=song_var, values=songs, width=50)
dropdown.pack(pady=10)

# Button to find the mood
find_button = tk.Button(root, text="Find Mood", font=("Arial", 12), bg="#4CAF50", fg="white", command=find_mood)
find_button.pack(pady=10)

# Label to display the result
result_label = tk.Label(root, text="", font=("Arial", 14), wraplength=400, height=5, width=40, bg='white',
                        relief="solid")
result_label.pack(pady=20)

root.geometry("800x400")
root.mainloop()
