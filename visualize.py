import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv('results/benchmark_v2.csv')
    
    # Separate custom_spec for plotting the curve
    spec_df = df[df['variant'] == 'custom_spec'].copy()
    spec_df['k'] = pd.to_numeric(spec_df['k'])
    spec_df = spec_df.sort_values('k')
    
    baseline_tps = df[df['variant'] == 'baseline']['tps'].values[0]
    hf_tps = df[df['variant'] == 'hf_assisted']['tps'].values[0]
    
    plt.figure(figsize=(10, 6))
    
    # Plot custom spec curve
    plt.plot(spec_df['k'], spec_df['speedup'], marker='o', linestyle='-', linewidth=2, label='Custom Speculative')
    
    # Highlight optimal K
    optimal_k = spec_df.loc[spec_df['speedup'].idxmax()]
    plt.scatter(optimal_k['k'], optimal_k['speedup'], color='red', s=100, zorder=5, label=f'Optimal (K={int(optimal_k["k"])})')
    
    # Horizontal line for Baseline
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1.0x)')
    
    # Point for HF Assisted
    plt.scatter(4, hf_tps/baseline_tps, color='green', marker='X', s=100, label='HF Assisted (auto K)')
    
    plt.title('Speculative Decoding Speedup vs. Lookahead (K)', fontsize=14)
    plt.xlabel('Lookahead (K)', fontsize=12)
    plt.ylabel('Speedup (Relative to Baseline)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/speedup_vs_k.png', dpi=300, bbox_inches='tight')
    print("Chart saved to docs/speedup_vs_k.png")

if __name__ == "__main__":
    main()
