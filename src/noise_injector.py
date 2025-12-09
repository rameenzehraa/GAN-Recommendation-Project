import pandas as pd
import numpy as np
import os

class NoiseInjector:
    def __init__(self, test_data):
        self.test_data = test_data.copy()
        
    def inject_random_noise(self, noise_level=0.05):
        """Inject random noise by changing ratings randomly"""
        noisy_data = self.test_data.copy()
        n_samples = len(noisy_data)
        n_noise = int(n_samples * noise_level)
        
        # Select random indices
        noise_indices = np.random.choice(n_samples, n_noise, replace=False)
        
        # Change ratings randomly (1-5)
        for idx in noise_indices:
            original_rating = noisy_data.iloc[idx]['rating']
            # Pick a different rating
            possible_ratings = [r for r in range(1, 6) if r != original_rating]
            noisy_data.iloc[idx, noisy_data.columns.get_loc('rating')] = np.random.choice(possible_ratings)
        
        return noisy_data
    
    def inject_shilling_attack(self, noise_level=0.05, target_movie_id=None):
        """Inject coordinated shilling attack (fake positive reviews)"""
        noisy_data = self.test_data.copy()
        n_samples = len(noisy_data)
        n_noise = int(n_samples * noise_level)
        
        # If no target specified, pick a random movie
        if target_movie_id is None:
            target_movie_id = np.random.choice(noisy_data['movie_id'].unique())
        
        # Create fake high ratings for target movie
        fake_ratings = []
        for _ in range(n_noise):
            fake_ratings.append({
                'user_id': np.random.choice(noisy_data['user_id'].unique()),
                'movie_id': target_movie_id,
                'rating': 5,  # Always 5 stars
                'timestamp': noisy_data['timestamp'].max() + 1
            })
        
        fake_df = pd.DataFrame(fake_ratings)
        noisy_data = pd.concat([noisy_data, fake_df], ignore_index=True)
        
        return noisy_data
    
    def create_all_noisy_versions(self, noise_levels=[0.05, 0.10, 0.15]):
        """Create all noisy test sets"""
        os.makedirs('data/noisy', exist_ok=True)
        
        for level in noise_levels:
            print(f"\nCreating {int(level*100)}% noisy test set...")
            
            # Random noise
            noisy_random = self.inject_random_noise(noise_level=level)
            filename = f'data/noisy/test_noise_{int(level*100)}pct.csv'
            noisy_random.to_csv(filename, index=False)
            print(f"✅ Saved: {filename}")
            
            # Shilling attack (only for 15%)
            if level == 0.15:
                noisy_shilling = self.inject_shilling_attack(noise_level=level/3)
                filename = f'data/noisy/test_shilling_{int(level*100)}pct.csv'
                noisy_shilling.to_csv(filename, index=False)
                print(f"✅ Saved: {filename}")

# Test the injector
if __name__ == "__main__":
    test_data = pd.read_csv('data/processed/test.csv')
    
    injector = NoiseInjector(test_data)
    injector.create_all_noisy_versions()
    
    print("\n✅ Noise injection complete!")