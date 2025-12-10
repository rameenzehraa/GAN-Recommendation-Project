import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# SETTINGS
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

RESULTS_CSV = 'results/metrics.csv'
ANALYSIS_TXT = 'results/written_analysis.txt'
SUMMARY_CSV = 'results/summary_table.csv'

# -----------------------------
# LOAD RESULTS
# -----------------------------
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"{RESULTS_CSV} not found.")

results = pd.read_csv(RESULTS_CSV)

print("="*70)
print("COMPLETE ANALYSIS: GAN vs COLLABORATIVE FILTERING UNDER NOISE")
print("="*70)
print("\nRaw Results:")
print(results)

# -----------------------------
# PERFORMANCE SUMMARY
# -----------------------------
metrics = ['precision', 'recall', 'ndcg', 'hit_rate']

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

for metric in metrics:
    cf_mean = results[f'cf_{metric}'].mean()
    gan_mean = results[f'gan_{metric}'].mean()
    winner = "GAN" if gan_mean > cf_mean else "CF"

    print(f"\n{metric.upper()}:")
    print(f"  CF:  {cf_mean:.4f} (average)")
    print(f"  GAN: {gan_mean:.4f} (average)")
    print(f"  Winner: {winner}")
    print("="*70)

# -----------------------------
# CLEAN DATA PERFORMANCE
# -----------------------------
print("CLEAN DATA PERFORMANCE (Baseline)")
print("="*70)

clean_results = results[results['test_set'] == 'test.csv'].iloc[0]

print("\nCollaborative Filtering:")
for metric in metrics:
    print(f"  {metric.capitalize()}@5: {clean_results[f'cf_{metric}']:.4f}")

print("\nGAN:")
for metric in metrics:
    print(f"  {metric.capitalize()}@5: {clean_results[f'gan_{metric}']:.4f}")

baseline_winner = "GAN" if clean_results['gan_precision'] > clean_results['cf_precision'] else "CF"
print(f"\n✅ {baseline_winner} performs better on clean data")
print("="*70)

# -----------------------------
# PERFORMANCE DEGRADATION
# -----------------------------
print("PERFORMANCE DEGRADATION ANALYSIS")
print("="*70)

clean_cf_precision = clean_results['cf_precision']
clean_gan_precision = clean_results['gan_precision']
clean_cf_ndcg = clean_results['cf_ndcg']
clean_gan_ndcg = clean_results['gan_ndcg']

degradation_data = []

for i in range(len(results)):
    if results['test_set'].iloc[i] == 'test.csv':
        continue  # Skip baseline

    noise_level = results['test_set'].iloc[i]

    cf_precision_drop = ((clean_cf_precision - results['cf_precision'].iloc[i]) / clean_cf_precision) * 100
    gan_precision_drop = ((clean_gan_precision - results['gan_precision'].iloc[i]) / clean_gan_precision) * 100
    cf_ndcg_drop = ((clean_cf_ndcg - results['cf_ndcg'].iloc[i]) / clean_cf_ndcg) * 100
    gan_ndcg_drop = ((clean_gan_ndcg - results['gan_ndcg'].iloc[i]) / clean_gan_ndcg) * 100

    print(f"\nNoise Level: {noise_level}")
    print(f"  Precision Drop: CF={cf_precision_drop:.2f}%, GAN={gan_precision_drop:.2f}% -> Winner={'GAN' if gan_precision_drop < cf_precision_drop else 'CF'}")
    print(f"  NDCG Drop:      CF={cf_ndcg_drop:.2f}%, GAN={gan_ndcg_drop:.2f}% -> Winner={'GAN' if gan_ndcg_drop < cf_ndcg_drop else 'CF'}")

    degradation_data.append({
        'noise_level': noise_level,
        'cf_precision_drop': cf_precision_drop,
        'gan_precision_drop': gan_precision_drop,
        'cf_ndcg_drop': cf_ndcg_drop,
        'gan_ndcg_drop': gan_ndcg_drop
    })

degradation_df = pd.DataFrame(degradation_data)

# -----------------------------
# ROBUSTNESS SCORE
# -----------------------------
print("\nROBUSTNESS SCORE (Lower is Better)")
print("="*70)

cf_avg_drop = degradation_df['cf_precision_drop'].mean()
gan_avg_drop = degradation_df['gan_precision_drop'].mean()
robustness_winner = "GAN" if gan_avg_drop < cf_avg_drop else "CF"
robustness_diff = abs(cf_avg_drop - gan_avg_drop)

print(f"Average Precision Drop: CF={cf_avg_drop:.2f}%, GAN={gan_avg_drop:.2f}%")
print(f"✅ {robustness_winner} is {robustness_diff:.2f}% more robust")
print("="*70)

# -----------------------------
# STATISTICAL ADVANTAGE
# -----------------------------
results['precision_diff'] = results['gan_precision'] - results['cf_precision']
results['ndcg_diff'] = results['gan_ndcg'] - results['cf_ndcg']

print("STATISTICAL ANALYSIS")
print("="*70)
for metric_diff, metric_name in zip(['precision_diff', 'ndcg_diff'], ['Precision', 'NDCG']):
    print(f"\n{metric_name} Advantage (GAN - CF):")
    for idx, row in results.iterrows():
        advantage = "GAN" if row[metric_diff] > 0 else "CF"
        print(f"  {row['test_set']:30s}: {row[metric_diff]:+0.4f} ({advantage} wins)")

print("="*70)

# -----------------------------
# SUMMARY TABLE & REPORT
# -----------------------------
summary_table = pd.DataFrame({
    'Test Set': results['test_set'],
    'CF Precision': results['cf_precision'].round(4),
    'GAN Precision': results['gan_precision'].round(4),
    'Winner': ['GAN' if g > c else 'CF' for g, c in zip(results['gan_precision'], results['cf_precision'])],
    'Performance Gap': (abs(results['gan_precision'] - results['cf_precision'])).round(4)
})

print("SUMMARY TABLE FOR REPORT")
print("="*70)
print(summary_table.to_string(index=False))

# Save summary table
summary_table.to_csv(SUMMARY_CSV, index=False)
print(f"\n✅ Summary table saved to: {SUMMARY_CSV}")

# -----------------------------
# WRITTEN ANALYSIS
# -----------------------------
analysis_text = f"""
ANALYSIS REPORT: GAN vs Collaborative Filtering Under Adversarial Noise

1. CLEAN DATA PERFORMANCE
{'='*60}
On clean test data, {baseline_winner} achieved superior Precision@5 of {max(clean_results['gan_precision'], clean_results['cf_precision']):.4f}.

2. PERFORMANCE UNDER NOISE
{'='*60}
CF degraded by {cf_avg_drop:.2f}%, GAN degraded by {gan_avg_drop:.2f}%.
Robustness winner: {robustness_winner} (maintains {robustness_diff:.2f}% more performance).

3. RECOMMENDATIONS
{'='*60}
Use {robustness_winner} in noisy/attack-prone environments.
Use {'GAN' if robustness_winner=='CF' else 'CF'} in clean/controlled environments.
"""

with open(ANALYSIS_TXT, 'w') as f:
    f.write(analysis_text)
print(f"\n✅ Analysis report saved to: {ANALYSIS_TXT}")
