import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import torch

class Evaluator:
    def __init__(self, cf_model_path, gan_model_path):
        """Load both models"""
        print("Loading models...")
        
        # Load CF model
        with open(cf_model_path, 'rb') as f:
            self.cf_model = pickle.load(f)
        
        # Load GAN model
        from gan_model import GANRecommender
        
        # Need to get num_users and num_items from checkpoint
        checkpoint = torch.load(gan_model_path, map_location='cpu')
        num_users = checkpoint['num_users']
        num_items = checkpoint['num_items']
        
        self.gan_model = GANRecommender(num_users, num_items)
        self.gan_model.load(gan_model_path)
        
        print("✅ Models loaded successfully!")
    
    def get_relevant_items(self, test_data, threshold=4.0):
        """Get relevant items per user (rating >= threshold)"""
        relevant = defaultdict(set)
        
        for _, row in test_data.iterrows():
            if row['rating'] >= threshold:
                relevant[row['user_id']].add(row['movie_id'])
        
        return relevant
    
    def precision_at_k(self, recommended, relevant, k=5):
        """Calculate Precision@K"""
        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])
        return relevant_count / k if k > 0 else 0
    
    def recall_at_k(self, recommended, relevant, k=5):
        """Calculate Recall@K"""
        if len(relevant) == 0:
            return 0
        recommended_k = recommended[:k]
        relevant_count = len([item for item in recommended_k if item in relevant])
        return relevant_count / len(relevant)
    
    def ndcg_at_k(self, recommended, relevant, k=5):
        """Calculate NDCG@K"""
        recommended_k = recommended[:k]
        
        # DCG
        dcg = 0
        for i, item in enumerate(recommended_k):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)
        
        # IDCG
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
        
        return dcg / idcg if idcg > 0 else 0
    
    def hit_rate_at_k(self, recommended, relevant, k=5):
        """Calculate Hit Rate@K"""
        recommended_k = recommended[:k]
        return 1 if any(item in relevant for item in recommended_k) else 0
    
    def evaluate_model(self, model, test_data, model_name="Model", k=5):
        """Evaluate a model on test data"""
        print(f"\nEvaluating {model_name}...")
        
        relevant_items = self.get_relevant_items(test_data)
        
        precisions = []
        recalls = []
        ndcgs = []
        hit_rates = []
        
        users_evaluated = 0
        
        for user_id in relevant_items.keys():
            if users_evaluated % 100 == 0:
                print(f"  Processed {users_evaluated} users...")
            
            try:
                # Get recommendations
                recommended = model.recommend(user_id, top_k=k)
                relevant = relevant_items[user_id]
                
                if len(recommended) > 0:
                    precisions.append(self.precision_at_k(recommended, relevant, k))
                    recalls.append(self.recall_at_k(recommended, relevant, k))
                    ndcgs.append(self.ndcg_at_k(recommended, relevant, k))
                    hit_rates.append(self.hit_rate_at_k(recommended, relevant, k))
                
                users_evaluated += 1
            except:
                continue
        
        results = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'ndcg': np.mean(ndcgs),
            'hit_rate': np.mean(hit_rates)
        }
        
        print(f"✅ {model_name} Evaluation Complete!")
        print(f"   Precision@{k}: {results['precision']:.4f}")
        print(f"   Recall@{k}: {results['recall']:.4f}")
        print(f"   NDCG@{k}: {results['ndcg']:.4f}")
        print(f"   Hit Rate@{k}: {results['hit_rate']:.4f}")
        
        return results
    
    def evaluate_all(self, test_files, k=5):
        """Evaluate both models on all test sets"""
        all_results = []
        
        for test_file in test_files:
            print("\n" + "="*60)
            print(f"📊 TESTING ON: {test_file}")
            print("="*60)
            
            test_data = pd.read_csv(test_file)
            
            # Evaluate CF
            cf_results = self.evaluate_model(self.cf_model, test_data, "Collaborative Filtering", k)
            
            # Evaluate GAN
            gan_results = self.evaluate_model(self.gan_model, test_data, "GAN", k)
            
            # Store results
            all_results.append({
                'test_set': test_file.split('/')[-1],
                'cf_precision': cf_results['precision'],
                'cf_recall': cf_results['recall'],
                'cf_ndcg': cf_results['ndcg'],
                'cf_hit_rate': cf_results['hit_rate'],
                'gan_precision': gan_results['precision'],
                'gan_recall': gan_results['recall'],
                'gan_ndcg': gan_results['ndcg'],
                'gan_hit_rate': gan_results['hit_rate']
            })
        
        return pd.DataFrame(all_results)

# Run evaluation
if __name__ == "__main__":
    import os
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator('models/cf_model.pkl', 'models/gan_model.pth')
    
    # Test files
    test_files = [
        'data/processed/test.csv',
        'data/noisy/test_noise_5pct.csv',
        'data/noisy/test_noise_10pct.csv',
        'data/noisy/test_noise_15pct.csv'
    ]
    
    # Run evaluation
    results_df = evaluator.evaluate_all(test_files, k=5)
    
    # Save results
    results_df.to_csv('results/metrics.csv', index=False)
    print("\n✅ Results saved to results/metrics.csv")
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))