"""
Professional Visualization for Experiment 2 Results
Generates publication-quality charts and graphs for LinkedIn/blog posts
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# Color palette
COLORS = {
    'gemma4': '#4285F4',    # Google Blue
    'gemma3': '#34A853',    # Google Green
    'mistral': '#EA4335',   # Red
    'accent': '#FBBC04',    # Yellow
    'grid': '#e0e0e0'
}

def load_results(filepath='exp2_results.json'):
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_overview_dashboard(data):
    """Create a comprehensive dashboard with all key metrics"""
    results = data['results']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Gemma 4 MoE Efficiency Benchmark - Complete Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Extract data
    models = [r['label'] for r in results]
    quality_scores = [r['avg_quality_score'] for r in results]
    quality_pcts = [r['quality_pct'] for r in results]
    speeds = [r['avg_tokens_per_sec'] for r in results]
    ram_usage = [r['memory']['ollama_model_ram_gb'] for r in results]
    params = [r['params'] for r in results]
    
    colors = [COLORS['gemma4'], COLORS['gemma3'], COLORS['mistral']]
    
    # 1. Quality Score Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(models, quality_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Average Quality Score', fontweight='bold')
    ax1.set_title('Quality Score (out of 3.0)', fontweight='bold', pad=10)
    ax1.set_xlim(0, 3.0)
    ax1.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score, pct) in enumerate(zip(bars1, quality_scores, quality_pcts)):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f} ({pct}%)', va='center', fontweight='bold')
    
    # 2. Speed Comparison (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(models, speeds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Tokens per Second', fontweight='bold')
    ax2.set_title('Inference Speed', fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, speed in zip(bars2, speeds):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{speed:.1f}', va='center', fontweight='bold')
    
    # 3. RAM Usage (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.barh(models, ram_usage, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('RAM Usage (GB)', fontweight='bold')
    ax3.set_title('Memory Footprint', fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, ram in zip(bars3, ram_usage):
        ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{ram:.1f} GB', va='center', fontweight='bold')
    
    # 4. Category Breakdown (Middle Left - spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    categories = list(results[0]['category_scores'].keys())
    x = np.arange(len(categories))
    width = 0.25
    
    for i, (result, color) in enumerate(zip(results, colors)):
        cat_scores = [result['category_scores'][cat] for cat in categories]
        offset = (i - 1) * width
        bars = ax4.bar(x + offset, cat_scores, width, label=result['label'], 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_ylabel('Score (out of 3.0)', fontweight='bold')
    ax4.set_title('Performance by Category', fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=0)
    ax4.set_ylim(0, 3.2)
    ax4.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Efficiency Scatter Plot (Middle Right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Create bubble sizes based on RAM (inverse - smaller RAM = bigger bubble)
    max_ram = max(ram_usage)
    bubble_sizes = [(max_ram - ram + 1) * 200 for ram in ram_usage]
    
    scatter = ax5.scatter(speeds, quality_scores, s=bubble_sizes, c=colors, 
                         alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, (model, x, y) in enumerate(zip(models, speeds, quality_scores)):
        ax5.annotate(model.split()[0] + ' ' + model.split()[1], 
                    (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    ax5.set_xlabel('Speed (tokens/sec)', fontweight='bold')
    ax5.set_ylabel('Quality Score', fontweight='bold')
    ax5.set_title('Quality vs Speed\n(bubble size = efficiency)', fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Model Architecture Comparison (Bottom Left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Show total params vs active params
    total_params = [float(p.replace('B', '')) for p in params]
    active_params = [float(r['active_params'].replace('B', '').replace('~', '')) for r in results]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax6.bar(x_pos - width/2, total_params, width, label='Total Params',
                   color='lightgray', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax6.bar(x_pos + width/2, active_params, width, label='Active Params',
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Parameters (Billions)', fontweight='bold')
    ax6.set_title('Model Size: Total vs Active', fontweight='bold', pad=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([m.split()[0] + '\n' + m.split()[1] for m in models], fontsize=9)
    ax6.legend(loc='upper left', framealpha=0.9)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}B', ha='center', va='bottom', fontsize=8)
    
    # 7. Efficiency Ratio (Bottom Middle)
    ax7 = fig.add_subplot(gs[2, 1])
    
    # Calculate efficiency: quality per GB of RAM
    efficiency = [q / r for q, r in zip(quality_scores, ram_usage)]
    
    bars = ax7.barh(models, efficiency, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax7.set_xlabel('Quality Score per GB RAM', fontweight='bold')
    ax7.set_title('Memory Efficiency', fontweight='bold', pad=10)
    ax7.grid(axis='x', alpha=0.3)
    
    for bar, eff in zip(bars, efficiency):
        ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{eff:.3f}', va='center', fontweight='bold')
    
    # 8. Key Insights Box (Bottom Right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate insights
    best_quality_idx = quality_scores.index(max(quality_scores))
    best_speed_idx = speeds.index(max(speeds))
    best_efficiency_idx = efficiency.index(max(efficiency))
    
    insights_text = f"""
    📊 KEY INSIGHTS
    
    🏆 Best Quality:
    {models[best_quality_idx]}
    {quality_scores[best_quality_idx]:.2f}/3.0 ({quality_pcts[best_quality_idx]}%)
    
    ⚡ Fastest:
    {models[best_speed_idx]}
    {speeds[best_speed_idx]:.1f} tok/s
    
    💾 Most Efficient:
    {models[best_efficiency_idx]}
    {efficiency[best_efficiency_idx]:.3f} quality/GB
    
    🎯 MoE Advantage:
    {"✅ Confirmed" if best_quality_idx == 0 or best_efficiency_idx == 0 else "⚠️ Mixed Results"}
    """
    
    ax8.text(0.1, 0.5, insights_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    plt.savefig('exp2_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: exp2_dashboard.png")
    
    return fig

def create_detailed_category_heatmap(data):
    """Create a heatmap showing performance across all categories"""
    results = data['results']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['label'] for r in results]
    categories = list(results[0]['category_scores'].keys())
    
    # Create matrix
    matrix = []
    for result in results:
        row = [result['category_scores'][cat] for cat in categories]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3.0)
    
    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score (out of 3.0)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Heatmap by Category', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig('exp2_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: exp2_heatmap.png")
    
    return fig

def create_radar_chart(data):
    """Create radar chart comparing models across categories"""
    results = data['results']
    
    categories = list(results[0]['category_scores'].keys())
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors_list = [COLORS['gemma4'], COLORS['gemma3'], COLORS['mistral']]
    
    for result, color in zip(results, colors_list):
        values = [result['category_scores'][cat] for cat in categories]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['label'], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticklabels(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], size=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Multi-Dimensional Performance Comparison', 
             size=14, fontweight='bold', pad=30)
    
    plt.savefig('exp2_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: exp2_radar.png")
    
    return fig

def create_linkedin_summary_card(data):
    """Create a clean summary card perfect for LinkedIn"""
    results = data['results']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    title_text = "Gemma 4 MoE Efficiency Benchmark Results"
    ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
           fontsize=20, fontweight='bold', ha='center')
    
    # Subtitle
    subtitle = f"Tested {len(results)} models × {data['prompts_count']} prompts = {len(results) * data['prompts_count']} total queries"
    ax.text(0.5, 0.90, subtitle, transform=ax.transAxes,
           fontsize=11, ha='center', style='italic', color='gray')
    
    # Create comparison table
    y_start = 0.80
    col_widths = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    headers = ['Model', 'Params', 'Active', 'Quality', 'Speed', 'RAM']
    
    # Draw headers
    x_pos = 0.05
    for header, width in zip(headers, col_widths):
        ax.text(x_pos, y_start, header, transform=ax.transAxes,
               fontsize=12, fontweight='bold', ha='left')
        x_pos += width
    
    # Draw separator line
    ax.plot([0.05, 0.95], [y_start - 0.02, y_start - 0.02], 
           'k-', linewidth=2, transform=ax.transAxes)
    
    # Draw data rows
    y_pos = y_start - 0.08
    colors_list = [COLORS['gemma4'], COLORS['gemma3'], COLORS['mistral']]
    
    for result, color in zip(results, colors_list):
        x_pos = 0.05
        
        # Model name (colored)
        ax.text(x_pos, y_pos, result['label'], transform=ax.transAxes,
               fontsize=11, ha='left', color=color, fontweight='bold')
        x_pos += col_widths[0]
        
        # Params
        ax.text(x_pos, y_pos, result['params'], transform=ax.transAxes,
               fontsize=11, ha='left')
        x_pos += col_widths[1]
        
        # Active
        ax.text(x_pos, y_pos, result['active_params'], transform=ax.transAxes,
               fontsize=11, ha='left')
        x_pos += col_widths[2]
        
        # Quality
        quality_text = f"{result['avg_quality_score']:.2f} ({result['quality_pct']}%)"
        ax.text(x_pos, y_pos, quality_text, transform=ax.transAxes,
               fontsize=11, ha='left', fontweight='bold')
        x_pos += col_widths[3]
        
        # Speed
        speed_text = f"{result['avg_tokens_per_sec']:.1f} t/s"
        ax.text(x_pos, y_pos, speed_text, transform=ax.transAxes,
               fontsize=11, ha='left')
        x_pos += col_widths[4]
        
        # RAM
        ram_text = f"{result['memory']['ollama_model_ram_gb']:.1f} GB"
        ax.text(x_pos, y_pos, ram_text, transform=ax.transAxes,
               fontsize=11, ha='left')
        
        y_pos -= 0.08
    
    # Footer
    timestamp = data.get('timestamp', 'N/A')
    footer_text = f"Generated: {timestamp[:10]} | Full results: exp2_results.json"
    ax.text(0.5, 0.02, footer_text, transform=ax.transAxes,
           fontsize=8, ha='center', color='gray', style='italic')
    
    plt.savefig('exp2_linkedin_card.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: exp2_linkedin_card.png")
    
    return fig

def main():
    print("🎨 Generating Professional Visualizations...")
    print("="*60)
    
    # Load data
    try:
        data = load_results()
        print(f"✅ Loaded results for {len(data['results'])} models")
    except FileNotFoundError:
        print("❌ Error: exp2_results.json not found!")
        print("   Run exp2_efficiency_test.py first to generate results.")
        return
    
    # Generate all visualizations
    print("\n📊 Creating visualizations...")
    
    create_overview_dashboard(data)
    create_detailed_category_heatmap(data)
    create_radar_chart(data)
    create_linkedin_summary_card(data)
    
    print("\n" + "="*60)
    print("✅ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. exp2_dashboard.png       - Comprehensive overview")
    print("  2. exp2_heatmap.png         - Category performance heatmap")
    print("  3. exp2_radar.png           - Multi-dimensional comparison")
    print("  4. exp2_linkedin_card.png   - Clean summary for social media")
    print("\n💡 Use these images in your LinkedIn post for maximum impact!")

if __name__ == "__main__":
    main()
