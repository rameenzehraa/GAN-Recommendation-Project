import pandas as pd
import os
from collaborative_filtering import CollaborativeFiltering
from gan_model import GANRecommender

def train_all_models():
    """Train both CF and GAN models"""
    print("="*50)
    print("TRAINING ALL MODELS")
    print("="*50)
    
    # Load data
    print("\n📂 Loading training data...")
    train_data = pd.read_csv('data/processed/train.csv')
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train Collaborative Filtering
    print("\n" + "="*50)
    print("1️⃣ TRAINING COLLABORATIVE FILTERING MODEL")
    print("="*50)
    
    cf = CollaborativeFiltering(n_factors=50)
    cf.fit(train_data, epochs=20, lr=0.01, reg=0.02)
    cf.save('models/cf_model.pkl')
    
    # Train GAN
    print("\n" + "="*50)
    print("2️⃣ TRAINING GAN MODEL")
    print("="*50)
    
    num_users = train_data['user_id'].nunique()
    num_items = train_data['movie_id'].nunique()
    
    gan = GANRecommender(num_users, num_items, embedding_dim=50)
    gan.train(train_data, epochs=50, batch_size=256)
    gan.save('models/gan_model.pth')
    
    print("\n" + "="*50)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*50)
    print("\nSaved models:")
    print("  - models/cf_model.pkl")
    print("  - models/gan_model.pth")

if __name__ == "__main__":
    train_all_models()