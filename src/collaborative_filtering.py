import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class CollaborativeFiltering:
    """
    Collaborative Filtering Recommender System using Matrix Factorization
    
    This is the TRADITIONAL approach being tested against the GAN model.
    It learns latent factors (hidden patterns) about users and items to predict ratings.
    
    How it works:
    - Decomposes the user-item rating matrix into two smaller matrices
    - User factors: What features/preferences each user has
    - Item factors: What features/characteristics each item has
    - Prediction = dot product of user and item factors + biases
    """
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_id_map = None
        self.item_id_map = None
        
    def fit(self, train_data, epochs=20, lr=0.01, reg=0.02):
        """Train using matrix factorization (SVD-like approach)"""
        print("Training Collaborative Filtering model...")
        
        # Create mappings
        unique_users = train_data['user_id'].unique()
        unique_items = train_data['movie_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = train_data['rating'].mean()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            for _, row in train_data.iterrows():
                user_idx = self.user_id_map[row['user_id']]
                item_idx = self.item_id_map[row['movie_id']]
                rating = row['rating']
                
                # Prediction
                pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
                pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                # Error
                error = rating - pred
                epoch_loss += error ** 2
                
                # Update biases
                self.user_bias[user_idx] += lr * (error - reg * self.user_bias[user_idx])
                self.item_bias[item_idx] += lr * (error - reg * self.item_bias[item_idx])
                
                # Update factors
                user_factor_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += lr * (error * self.item_factors[item_idx] - reg * self.user_factors[user_idx])
                self.item_factors[item_idx] += lr * (error * user_factor_old - reg * self.item_factors[item_idx])
            
            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(epoch_loss / len(train_data))
                print(f"Epoch {epoch+1}/{epochs}, RMSE: {rmse:.4f}")
        
        print("✅ CF model training complete!")
    
    def predict(self, user_id, movie_id):
        """Predict rating for user-movie pair"""
        if user_id not in self.user_id_map or movie_id not in self.item_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[movie_id]
        
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return np.clip(pred, 1, 5)
    
    def recommend(self, user_id, top_k=5):
        """Recommend top-K items for user"""
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Predict for all items
        scores = []
        for movie_id, item_idx in self.item_id_map.items():
            score = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
            score += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            scores.append((movie_id, score))
        
        # Sort and return top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in scores[:top_k]]
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Test the model
if __name__ == "__main__":
    train_data = pd.read_csv('data/processed/train.csv')
    
    cf = CollaborativeFiltering(n_factors=50)
    cf.fit(train_data, epochs=10)
    
    # Test prediction
    test_user = train_data['user_id'].iloc[0]
    test_movie = train_data['movie_id'].iloc[0]
    pred = cf.predict(test_user, test_movie)
    print(f"\nTest prediction: User {test_user}, Movie {test_movie} -> Rating: {pred:.2f}")
    
    # Save model
    cf.save('models/cf_model.pkl')