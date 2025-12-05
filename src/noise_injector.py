import pandas as pd
import numpy as np
import random

class NoiseInjector:
    """Injects different types of noise into rating data"""
    
    def __init__(self, ratings_df):
        self.original_ratings = ratings_df.copy()
        self.noisy_ratings = None
        
    def inject_random_noise(self, noise_percentage=5):
        """
        Inject random wrong ratings
        Args:
            noise_percentage: percentage of ratings to corrupt (5, 10, or 15)
        """
        print(f"\n=== Injecting {noise_percentage}% Random Noise ===")
        
        self.noisy_ratings = self.original_ratings.copy()
        n_ratings = len(self.noisy_ratings)
        n_corrupt = int(n_ratings * (noise_percentage / 100))
        
        # Randomly select ratings to corrupt
        corrupt_indices = random.sample(range(n_ratings), n_corrupt)
        
        # Replace with random ratings (1-5)
        for idx in corrupt_indices:
            original_rating = self.noisy_ratings.loc[idx, 'rating']
            # Pick a different random rating
            new_rating = random.choice([r for r in [1, 2, 3, 4, 5] if r != original_rating])
            self.noisy_ratings.loc[idx, 'rating'] = new_rating
        
        print(f"✓ Corrupted {n_corrupt} ratings out of {n_ratings}")
        return self.noisy_ratings
    
    def inject_shilling_attack(self, noise_percentage=5, attack_users=10):
        """
        Inject coordinated shilling attacks (malicious users promoting/demoting items)
        Args:
            noise_percentage: percentage of ratings affected
            attack_users: number of fake attack users to create
        """
        print(f"\n=== Injecting {noise_percentage}% Shilling Attack ===")
        
        self.noisy_ratings = self.original_ratings.copy()
        
        # Get max user_id to create fake users
        max_user_id = self.noisy_ratings['user_id'].max()
        
        # Select random movies to attack (promote with 5 stars)
        unique_movies = self.noisy_ratings['movie_id'].unique()
        n_target_movies = max(1, int(len(unique_movies) * (noise_percentage / 100)))
        target_movies = random.sample(list(unique_movies), n_target_movies)
        
        # Create fake users that give target movies all 5-star ratings
        fake_ratings = []
        for i in range(attack_users):
            fake_user_id = max_user_id + i + 1
            for movie_id in target_movies:
                fake_ratings.append({
                    'user_id': fake_user_id,
                    'movie_id': movie_id,
                    'rating': 5,  # Always 5 stars (push attack)
                    'timestamp': 0
                })
        
        # Add fake ratings to dataset
        fake_df = pd.DataFrame(fake_ratings)
        self.noisy_ratings = pd.concat([self.noisy_ratings, fake_df], ignore_index=True)
        
        print(f"✓ Added {len(fake_ratings)} fake ratings from {attack_users} attack users")
        print(f"✓ Targeting {n_target_movies} movies for promotion")
        return self.noisy_ratings
    
    def inject_mixed_noise(self, noise_percentage=10):
        """
        Mix of random noise + shilling attack
        """
        print(f"\n=== Injecting {noise_percentage}% Mixed Noise ===")
        
        # First inject random noise (half the percentage)
        self.inject_random_noise(noise_percentage // 2)
        
        # Then inject shilling attack (other half)
        self.inject_shilling_attack(noise_percentage // 2, attack_users=5)
        
        return self.noisy_ratings
    
    def get_noise_statistics(self):
        """Compare original vs noisy data"""
        if self.noisy_ratings is None:
            print("No noise injected yet!")
            return
        
        print("\n=== Noise Impact Statistics ===")
        print(f"Original ratings: {len(self.original_ratings)}")
        print(f"Noisy ratings: {len(self.noisy_ratings)}")
        print(f"Added ratings: {len(self.noisy_ratings) - len(self.original_ratings)}")
        print(f"Original avg rating: {self.original_ratings['rating'].mean():.2f}")
        print(f"Noisy avg rating: {self.noisy_ratings['rating'].mean():.2f}")
        print(f"Rating shift: {abs(self.original_ratings['rating'].mean() - self.noisy_ratings['rating'].mean()):.2f}")


# Test the noise injector
if __name__ == "__main__":
    from data_loader import MovieLensLoader
    
    # Load data
    loader = MovieLensLoader()
    ratings, _, _ = loader.load_data()
    
    # Test different noise types
    print("\n" + "="*50)
    print("Testing 5% Random Noise")
    print("="*50)
    injector1 = NoiseInjector(ratings)
    noisy_5 = injector1.inject_random_noise(5)
    injector1.get_noise_statistics()
    
    print("\n" + "="*50)
    print("Testing 10% Shilling Attack")
    print("="*50)
    injector2 = NoiseInjector(ratings)
    noisy_10 = injector2.inject_shilling_attack(10, attack_users=15)
    injector2.get_noise_statistics()
    
    print("\n" + "="*50)
    print("Testing 15% Mixed Noise")
    print("="*50)
    injector3 = NoiseInjector(ratings)
    noisy_15 = injector3.inject_mixed_noise(15)
    injector3.get_noise_statistics()