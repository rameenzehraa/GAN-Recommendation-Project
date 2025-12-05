import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MovieLensLoader:
    """Loads and prepares MovieLens-1M dataset"""
    
    def __init__(self, data_path='data/ml-1m'):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_data(self):
        """Load all dataset files"""
        print("Loading MovieLens-1M dataset...")
        
        # Load ratings (UserID::MovieID::Rating::Timestamp)
        self.ratings = pd.read_csv(
            f'{self.data_path}/ratings.dat',
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load movies (MovieID::Title::Genres)
        self.movies = pd.read_csv(
            f'{self.data_path}/movies.dat',
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )
        
        # Load users (UserID::Gender::Age::Occupation::Zip-code)
        self.users = pd.read_csv(
            f'{self.data_path}/users.dat',
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            encoding='latin-1'
        )
        
        print(f"✓ Loaded {len(self.ratings)} ratings")
        print(f"✓ Loaded {len(self.movies)} movies")
        print(f"✓ Loaded {len(self.users)} users")
        
        return self.ratings, self.movies, self.users
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split ratings into train and test sets"""
        if self.ratings is None:
            self.load_data()
        
        train_data, test_data = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"✓ Train set: {len(train_data)} ratings")
        print(f"✓ Test set: {len(test_data)} ratings")
        
        return train_data, test_data
    
    def get_statistics(self):
        """Print dataset statistics"""
        if self.ratings is None:
            self.load_data()
        
        print("\n=== Dataset Statistics ===")
        print(f"Total ratings: {len(self.ratings)}")
        print(f"Unique users: {self.ratings['user_id'].nunique()}")
        print(f"Unique movies: {self.ratings['movie_id'].nunique()}")
        print(f"Rating range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"Average rating: {self.ratings['rating'].mean():.2f}")
        print(f"Sparsity: {(1 - len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['movie_id'].nunique())) * 100:.2f}%")


# Test the loader
if __name__ == "__main__":
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_data()
    loader.get_statistics()
    train, test = loader.get_train_test_split()