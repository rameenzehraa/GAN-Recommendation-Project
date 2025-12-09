import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_comparison_charts():
    """Create all visualization charts"""
    print("Creating visualizations...")
    
    # Load results
    results = pd.read_csv('results/metrics.csv')
    
    # Create charts directory
    os.makedirs('results/charts', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Precision Comparison
    fig, ax = plt.subplots()
    x = range(len(results))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], results['cf_precision'], width, label='CF', color='#3498db')
    ax.bar([i + width/2 for i in x], results['gan_precision'], width, label='GAN', color='#e74c3c')
    
    ax.set_xlabel('Test Set')
    ax.set_ylabel('Precision@5')
    ax.set_title('Precision@5 Comparison: CF vs GAN')
    ax.set_xticks(x)
    ax.set_xticklabels(results['test_set'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/precision_comparison.png', dpi=300)
    print("✅ Saved: precision_comparison.png")
    plt.close()
    
    # 2. NDCG Comparison
    fig, ax = plt.subplots()
    
    ax.bar([i - width/2 for i in x], results['cf_ndcg'], width, label='CF', color='#3498db')
    ax.bar([i + width/2 for i in x], results['gan_ndcg'], width, label='GAN', color='#e74c3c')
    
    ax.set_xlabel('Test Set')
    ax.set_ylabel('NDCG@5')
    ax.set_title('NDCG@5 Comparison: CF vs GAN')
    ax.set_xticks(x)
    ax.set_xticklabels(results['test_set'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/ndcg_comparison.png', dpi=300)
    print("✅ Saved: ndcg_comparison.png")
    plt.close()
    
    # 3. Degradation Curves
    fig, ax = plt.subplots()
    
    noise_levels = ['Clean', '5%', '10%', '15%']
    
    ax.plot(noise_levels, results['cf_precision'], marker='o', linewidth=2, markersize=8, label='CF', color='#3498db')
    ax.plot(noise_levels, results['gan_precision'], marker='s', linewidth=2, markersize=8, label='GAN', color='#e74c3c')
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Precision@5')
    ax.set_title('Performance Degradation Under Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/degradation_curves.png', dpi=300)
    print("✅ Saved: degradation_curves.png")
    plt.close()
    
    # 4. All Metrics Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    heatmap_data = pd.DataFrame({
        'CF Precision': results['cf_precision'],
        'GAN Precision': results['gan_precision'],
        'CF Recall': results['cf_recall'],
        'GAN Recall': results['gan_recall'],
        'CF NDCG': results['cf_ndcg'],
        'GAN NDCG': results['gan_ndcg'],
        'CF Hit Rate': results['cf_hit_rate'],
        'GAN Hit Rate': results['gan_hit_rate']
    })
    heatmap_data.index = results['test_set']
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, ax=ax)
    ax.set_title('All Metrics Heatmap')
    
    plt.tight_layout()
    plt.savefig('results/charts/metrics_heatmap.png', dpi=300)
    print("✅ Saved: metrics_heatmap.png")
    plt.close()
    
    # 5. Robustness Score (percent performance retained)
    fig, ax = plt.subplots()
    
    cf_retention = (results['cf_precision'].iloc[1:] / results['cf_precision'].iloc[0]) * 100
    gan_retention = (results['gan_precision'].iloc[1:] / results['gan_precision'].iloc[0]) * 100
    
    noise_labels = ['5%', '10%', '15%']
    
    ax.bar([i - width/2 for i in range(len(noise_labels))], cf_retention, width, label='CF', color='#3498db')
    ax.bar([i + width/2 for i in range(len(noise_labels))], gan_retention, width, label='GAN', color='#e74c3c')
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Performance Retained (%)')
    ax.set_title('Robustness: Performance Retention Under Noise')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/robustness_score.png', dpi=300)
    print("✅ Saved: robustness_score.png")
    plt.close()
    
    print("\n✅ All visualizations created in results/charts/")

if __name__ == "__main__":
    create_comparison_charts()