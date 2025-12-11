import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class MovieLensLoader:
    """
    Loads and prepares the MovieLens-1M dataset for training recommendation systems.
    
    MovieLens-1M contains:
    - 1 million movie ratings from 6,000 users on 4,000 movies
    - User demographics (age, gender, occupation)
    - Movie metadata (title, genres)
    """
    def __init__(self, data_path='data/raw/ml-1m'):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_data(self):
        """
        Load all three files from the MovieLens dataset.
        
        The files use '::' as separators and have no headers, so we manually
        specify column names.
        
        Returns:
            Tuple of (ratings, movies, users) DataFrames
        """

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
        
        # Print summary statistics to verify data loaded correctly
        print(f"Loaded {len(self.ratings)} ratings")
        print(f"Users: {self.ratings['user_id'].nunique()}")
        print(f"Movies: {self.ratings['movie_id'].nunique()}")
        
        return self.ratings, self.movies, self.users
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split ratings into training and testing sets.
        
        Training set: Used to train both CF and GAN models (80% of data)
        Testing set: Used to evaluate model performance (20% of data)
        
        Args:
            test_size: Fraction of data to use for testing (default 20%)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        
        if self.ratings is None:
            self.load_data()
        
        train_data, test_data = train_test_split(
            self.ratings, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Save processed data to CSV files for later use
        os.makedirs('data/processed', exist_ok=True)
        train_data.to_csv('data/processed/train.csv', index=False)
        test_data.to_csv('data/processed/test.csv', index=False)
        
        print(f"\nTrain set: {len(train_data)} ratings")
        print(f"Test set: {len(test_data)} ratings")
        
        return train_data, test_data
    
    def get_user_item_matrix(self, data):
        """
        Convert ratings data into a user-item matrix.
        
        This creates a 2D table where:
        - Rows = users
        - Columns = movies
        - Values = ratings (0 if user hasn't rated that movie)
        
        Args:
            data: DataFrame with user_id, movie_id, rating columns
            
        Returns:
            DataFrame in matrix format (users x movies)
        """
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