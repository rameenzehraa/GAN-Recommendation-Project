import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class MovieLensLoader:
    def __init__(self, data_path='data/raw/ml-1m'):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_data(self):
        """Load MovieLens-1M dataset"""
        print("Loading MovieLens-1M dataset...")
        
        # Load ratings
        self.ratings = pd.read_csv(
            f'{self.data_path}/ratings.dat',
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load movies
        self.movies = pd.read_csv(
            f'{self.data_path}/movies.dat',
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )
        
        # Load users
        self.users = pd.read_csv(
            f'{self.data_path}/users.dat',
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            encoding='latin-1'
        )
        
        print(f"Loaded {len(self.ratings)} ratings")
        print(f"Users: {self.ratings['user_id'].nunique()}")
        print(f"Movies: {self.ratings['movie_id'].nunique()}")
        
        return self.ratings, self.movies, self.users
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if self.ratings is None:
            self.load_data()
        
        train_data, test_data = train_test_split(
            self.ratings, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Save to processed folder
        os.makedirs('data/processed', exist_ok=True)
        train_data.to_csv('data/processed/train.csv', index=False)
        test_data.to_csv('data/processed/test.csv', index=False)
        
        print(f"\nTrain set: {len(train_data)} ratings")
        print(f"Test set: {len(test_data)} ratings")
        
        return train_data, test_data
    
    def get_user_item_matrix(self, data):
        """Create user-item rating matrix"""
        matrix = data.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        return matrix

# Test the loader
if __name__ == "__main__":
    loader = MovieLensLoader()
    train, test = loader.get_train_test_split()
    print("\n✅ Data loader working correctly!")