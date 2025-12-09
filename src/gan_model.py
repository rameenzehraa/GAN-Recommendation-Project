import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    def __init__(self, data):
        self.users = torch.LongTensor(data['user_id'].values)
        self.items = torch.LongTensor(data['movie_id'].values)
        self.ratings = torch.FloatTensor(data['rating'].values)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class Generator(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[256, 128, 64]):
        super(Generator, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=1)
        output = self.mlp(x)
        
        # Scale to rating range (1-5)
        rating = output * 4 + 1
        return rating.squeeze()

class Discriminator(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[256, 128, 64]):
        super(Discriminator, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP (includes rating as input)
        layers = []
        input_dim = embedding_dim * 2 + 1  # +1 for rating
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Probability output
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids, ratings):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Normalize rating to 0-1 range
        rating_normalized = (ratings - 1) / 4
        rating_input = rating_normalized.unsqueeze(1)
        
        x = torch.cat([user_emb, item_emb, rating_input], dim=1)
        output = self.mlp(x)
        
        return output.squeeze()

class GANRecommender:
    def __init__(self, num_users, num_items, embedding_dim=50, device='cpu'):
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        
        self.generator = Generator(num_users, num_items, embedding_dim).to(device)
        self.discriminator = Discriminator(num_users, num_items, embedding_dim).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        
        self.criterion = nn.BCELoss()
        
        # ID mappings
        self.user_id_map = None
        self.item_id_map = None
        
    def create_mappings(self, data):
        """Create user/item ID to index mappings"""
        unique_users = sorted(data['user_id'].unique())
        unique_items = sorted(data['movie_id'].unique())
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
    def map_ids(self, data):
        """Map original IDs to indices"""
        mapped_data = data.copy()
        mapped_data['user_id'] = mapped_data['user_id'].map(self.user_id_map)
        mapped_data['movie_id'] = mapped_data['movie_id'].map(self.item_id_map)
        return mapped_data.dropna()
    
    def train(self, train_data, epochs=50, batch_size=256):
        """Train GAN"""
        print("Training GAN model...")
        
        # Create mappings
        self.create_mappings(train_data)
        mapped_data = self.map_ids(train_data)
        
        dataset = RatingDataset(mapped_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            g_loss_total = 0
            d_loss_total = 0
            
            for batch_idx, (users, items, ratings) in enumerate(dataloader):
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                
                batch_size_actual = users.size(0)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_output = self.discriminator(users, items, ratings)
                real_labels = torch.ones(batch_size_actual).to(self.device)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake data
                fake_ratings = self.generator(users, items)
                fake_output = self.discriminator(users, items, fake_ratings.detach())
                fake_labels = torch.zeros(batch_size_actual).to(self.device)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_ratings = self.generator(users, items)
                fake_output = self.discriminator(users, items, fake_ratings)
                g_loss = self.criterion(fake_output, real_labels)  # Want discriminator to think it's real
                
                g_loss.backward()
                self.g_optimizer.step()
                
                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - G_Loss: {g_loss_total/len(dataloader):.4f}, D_Loss: {d_loss_total/len(dataloader):.4f}")
        
        print("✅ GAN training complete!")
    
    def predict(self, user_id, movie_id):
        """Predict rating"""
        if user_id not in self.user_id_map or movie_id not in self.item_id_map:
            return 3.0  # Default rating
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[movie_id]
        
        self.generator.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            rating = self.generator(user_tensor, item_tensor)
        
        return rating.item()
    
    def recommend(self, user_id, top_k=5):
        """Recommend top-K items"""
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        self.generator.eval()
        scores = []
        
        with torch.no_grad():
            for movie_id, item_idx in self.item_id_map.items():
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                score = self.generator(user_tensor, item_tensor).item()
                scores.append((movie_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in scores[:top_k]]
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'num_users': self.num_users,
            'num_items': self.num_items
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.user_id_map = checkpoint['user_id_map']
        self.item_id_map = checkpoint['item_id_map']
        print(f"✅ Model loaded from {filepath}")

# Test the GAN
if __name__ == "__main__":
    train_data = pd.read_csv('data/processed/train.csv')
    
    num_users = train_data['user_id'].nunique()
    num_items = train_data['movie_id'].nunique()
    
    gan = GANRecommender(num_users, num_items, embedding_dim=50)
    gan.train(train_data, epochs=10, batch_size=256)
    
    # Save model
    gan.save('models/gan_model.pth')