# visualization_suite.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class RCRVisualizer:
    """Visualization suite for RCR Router analytics"""
    
    def __init__(self, analytics_data: Optional[Dict] = None):
        self.analytics_data = analytics_data or {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            "broadcast": "#e74c3c",  # Red
            "rcr": "#2ecc71",        # Green  
            "planner": "#3498db",    # Blue
            "coder": "#f39c12",      # Orange
            "reviewer": "#9b59b6",   # Purple
            "memory": "#95a5a6",     # Gray
            "router": "#e67e22"      # Dark orange
        }
    
    def create_router_architecture_diagram(self, save_path: str = "rcr_architecture.png"):
        """Create a visual representation of the RCR router architecture"""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        positions = {
            # Memory layer
            "Shared Memory": (2, 4),
            "Memory Items": (1, 3.5),
            "Conflict Resolution": (3, 3.5),
            
            # Router core
            "RCR Router": (2, 2.5),
            "Token Budget": (0.5, 2),
            "Importance Scorer": (2, 2),
            "Semantic Filter": (3.5, 2),
            
            # Agents
            "Planner Agent": (0.5, 0.5),
            "Coder Agent": (2, 0.5),
            "Reviewer Agent": (3.5, 0.5),
            
            # Providers
            "Gemini": (0.5, -0.5),
            "Groq": (2, -0.5),
            "Ollama": (3.5, -0.5),
        }
        
        # Add nodes
        memory_nodes = ["Shared Memory", "Memory Items", "Conflict Resolution"]
        router_nodes = ["RCR Router", "Token Budget", "Importance Scorer", "Semantic Filter"]
        agent_nodes = ["Planner Agent", "Coder Agent", "Reviewer Agent"]
        provider_nodes = ["Gemini", "Groq", "Ollama"]
        
        G.add_nodes_from(memory_nodes)
        G.add_nodes_from(router_nodes)
        G.add_nodes_from(agent_nodes)
        G.add_nodes_from(provider_nodes)
        
        # Add edges (information flow)
        edges = [
            # Memory to Router
            ("Shared Memory", "RCR Router"),
            ("Memory Items", "Importance Scorer"),
            ("Conflict Resolution", "Shared Memory"),
            
            # Router internal flow
            ("RCR Router", "Token Budget"),
            ("RCR Router", "Importance Scorer"), 
            ("RCR Router", "Semantic Filter"),
            ("Token Budget", "Semantic Filter"),
            ("Importance Scorer", "Semantic Filter"),
            
            # Router to Agents
            ("Semantic Filter", "Planner Agent"),
            ("Semantic Filter", "Coder Agent"),
            ("Semantic Filter", "Reviewer Agent"),
            
            # Agents to Providers
            ("Planner Agent", "Gemini"),
            ("Coder Agent", "Groq"),
            ("Reviewer Agent", "Ollama"),
            
            # Feedback loop (Agents back to Memory)
            ("Planner Agent", "Conflict Resolution"),
            ("Coder Agent", "Conflict Resolution"),
            ("Reviewer Agent", "Conflict Resolution"),
        ]
        
        G.add_edges_from(edges)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, positions, nodelist=memory_nodes, 
                              node_color=self.colors["memory"], node_size=1500, alpha=0.8)
        nx.draw_networkx_nodes(G, positions, nodelist=router_nodes,
                              node_color=self.colors["router"], node_size=1200, alpha=0.8)
        nx.draw_networkx_nodes(G, positions, nodelist=agent_nodes,
                              node_color=[self.colors["planner"], self.colors["coder"], self.colors["reviewer"]], 
                              node_size=1000, alpha=0.8)
        nx.draw_networkx_nodes(G, positions, nodelist=provider_nodes,
                              node_color="lightblue", node_size=800, alpha=0.6)
        
        # Draw edges with different styles
        memory_edges = [e for e in edges if e[0] in memory_nodes or e[1] in memory_nodes]
        router_edges = [e for e in edges if e[0] in router_nodes and e[1] in router_nodes]
        flow_edges = [e for e in edges if e not in memory_edges and e not in router_edges]
        
        nx.draw_networkx_edges(G, positions, edgelist=memory_edges, 
                              edge_color=self.colors["memory"], width=2, alpha=0.7)
        nx.draw_networkx_edges(G, positions, edgelist=router_edges,
                              edge_color=self.colors["router"], width=2, alpha=0.7) 
        nx.draw_networkx_edges(G, positions, edgelist=flow_edges,
                              edge_color="gray", width=1, alpha=0.5, style="dashed")
        
        # Add labels
        nx.draw_networkx_labels(G, positions, font_size=8, font_weight="bold")
        
        # Add title and annotations
        plt.title("RCR Router Architecture\nRole-Aware Context Routing with Structured Memory", 
                 fontsize=16, fontweight="bold", pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["memory"], 
                      markersize=12, label='Memory Layer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["router"], 
                      markersize=12, label='Router Core'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["planner"], 
                      markersize=12, label='Agents'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="lightblue", 
                      markersize=12, label='LLM Providers'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Architecture diagram saved to {save_path}")
    
    def create_token_savings_chart(self, comparison_data: pd.DataFrame, save_path: str = "token_savings.png"):
        """Create bar chart showing token savings per role/metric"""
        
        if comparison_data.empty:
            print("No comparison data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract numeric values for plotting
        metrics = []
        broadcast_vals = []
        rcr_vals = []
        improvements = []
        
        for _, row in comparison_data.iterrows():
            metric = row['Metric']
            
            # Skip cost for now (has $ symbol)
            if 'Cost' in metric:
                continue
                
            broadcast = row['Broadcast All']
            rcr = row['Role-Aware Routing'] 
            improvement = row['Improvement']
            
            # Extract numeric values
            if isinstance(broadcast, str):
                broadcast_num = float(broadcast.replace('ms', '').replace('$', ''))
            else:
                broadcast_num = float(broadcast)
                
            if isinstance(rcr, str):
                rcr_num = float(rcr.replace('ms', '').replace('$', ''))
            else:
                rcr_num = float(rcr)
            
            metrics.append(metric)
            broadcast_vals.append(broadcast_num)
            rcr_vals.append(rcr_num)
            
            # Extract improvement percentage
            if isinstance(improvement, str) and '%' in improvement:
                imp_num = float(improvement.replace('%', '').replace('-', '').replace('+', ''))
                improvements.append(imp_num)
            else:
                improvements.append(0)
        
        # Plot 1: Absolute values comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, broadcast_vals, width, label='Broadcast All', 
                       color=self.colors["broadcast"], alpha=0.8)
        bars2 = ax1.bar(x + width/2, rcr_vals, width, label='Role-Aware Routing',
                       color=self.colors["rcr"], alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('Broadcast All vs Role-Aware Routing')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Improvement percentages
        bars3 = ax2.bar(metrics, improvements, color=self.colors["rcr"], alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Percentage Improvements with RCR Router')
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Token savings chart saved to {save_path}")
    
    def create_per_agent_performance_dashboard(self, agent_data: pd.DataFrame, save_path: str = "agent_performance.png"):
        """Create comprehensive per-agent performance dashboard"""
        
        if agent_data.empty:
            print("No agent data available")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        agents = agent_data['Agent'].tolist()
        roles = agent_data['Role'].tolist()
        
        # Plot 1: Tokens per Agent
        tokens = agent_data['Total Tokens'].tolist()
        role_colors = [self.colors.get(role.lower(), 'gray') for role in roles]
        
        bars1 = ax1.bar(agents, tokens, color=role_colors, alpha=0.8)
        ax1.set_title('Total Tokens by Agent', fontweight='bold')
        ax1.set_ylabel('Total Tokens')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, token_count in zip(bars1, tokens):
            ax1.annotate(f'{token_count:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Average Latency
        latencies = agent_data['Avg Latency (ms)'].tolist()
        bars2 = ax2.bar(agents, latencies, color=role_colors, alpha=0.8)
        ax2.set_title('Average Latency by Agent', fontweight='bold')
        ax2.set_ylabel('Latency (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, latency in zip(bars2, latencies):
            ax2.annotate(f'{latency:.0f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Quality Scores
        quality_scores = agent_data['Avg Quality'].tolist()
        bars3 = ax3.bar(agents, quality_scores, color=role_colors, alpha=0.8)
        ax3.set_title('Average Quality Score by Agent', fontweight='bold')
        ax3.set_ylabel('Quality Score (1-5)')
        ax3.set_ylim(0, 5)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars3, quality_scores):
            ax3.annotate(f'{score:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Context Efficiency
        context_eff = [float(x.replace('%', '')) for x in agent_data['Context Efficiency'].tolist()]
        bars4 = ax4.bar(agents, context_eff, color=role_colors, alpha=0.8)
        ax4.set_title('Context Efficiency by Agent', fontweight='bold')
        ax4.set_ylabel('Efficiency (%)')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars4, context_eff):
            ax4.annotate(f'{eff:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Add overall title
        fig.suptitle('RCR Router: Per-Agent Performance Dashboard', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Agent performance dashboard saved to {save_path}")
    
    def create_interactive_timeline(self, turn_metrics: List[Dict], save_path: str = "interactive_timeline.html"):
        """Create interactive timeline of agent turns with Plotly"""
        
        if not turn_metrics:
            print("No turn metrics available")
            return
        
        # Prepare data
        df = pd.DataFrame(turn_metrics)
        df['start_datetime'] = pd.to_datetime(df['start_time'], unit='s')
        
        # Create timeline plot
        fig = px.timeline(
            df, 
            x_start='start_datetime', 
            x_end='end_datetime' if 'end_datetime' in df.columns else 'start_datetime',
            y='agent_name',
            color='role',
            title="RCR Router: Agent Execution Timeline",
            labels={'agent_name': 'Agent', 'role': 'Role'},
            hover_data=['total_tokens', 'provider_used', 'quality_score', 'total_latency_ms']
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Agent"
        )
        
        fig.write_html(save_path)
        print(f"Interactive timeline saved to {save_path}")
    
    def create_memory_evolution_chart(self, memory_stats: List[Dict], save_path: str = "memory_evolution.png"):
        """Chart showing how memory evolves over time"""
        
        if not memory_stats:
            print("No memory evolution data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        rounds = list(range(1, len(memory_stats) + 1))
        total_items = [stats.get('total_items', 0) for stats in memory_stats]
        avg_confidence = [stats.get('avg_confidence', 0) for stats in memory_stats]
        
        # Plot 1: Memory items over time
        ax1.plot(rounds, total_items, marker='o', linewidth=2, markersize=6, 
                color=self.colors['memory'])
        ax1.set_title('Memory Items Growth Over Time', fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Total Memory Items')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average confidence over time
        ax2.plot(rounds, avg_confidence, marker='s', linewidth=2, markersize=6,
                color=self.colors['router'])
        ax2.set_title('Average Memory Confidence Over Time', fontweight='bold')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Average Confidence')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Memory evolution chart saved to {save_path}")

def generate_ready_report(
    analytics: 'RCRAnalytics',
    comparison_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    output_dir: str = "report"
):
    """Generate visualizations and summary"""
    
    from pathlib import Path
    Path(output_dir).mkdir(exist_ok=True)
    
    visualizer = RCRVisualizer()
    
    # 1. Architecture diagram
    visualizer.create_router_architecture_diagram(f"{output_dir}/rcr_architecture.png")
    
    # 2. Token savings comparison
    visualizer.create_token_savings_chart(comparison_df, f"{output_dir}/token_savings.png")
    
    # 3. Per-agent performance
    if not agent_df.empty:
        visualizer.create_per_agent_performance_dashboard(agent_df, f"{output_dir}/agent_performance.png")
    
    # 4. Create summary infographic data
    if not comparison_df.empty:
        # Extract key metrics
        total_tokens_row = comparison_df[comparison_df['Metric'] == 'Total Tokens']
        if not total_tokens_row.empty:
            improvement = total_tokens_row['Improvement'].iloc[0]
            token_savings = improvement.replace('-', '').replace('%', '')
        else:
            token_savings = "N/A"
        
        context_tokens_row = comparison_df[comparison_df['Metric'] == 'Context Tokens']
        if not context_tokens_row.empty:
            context_improvement = context_tokens_row['Improvement'].iloc[0]
            context_savings = context_improvement.replace('-', '').replace('%', '')
        else:
            context_savings = "N/A"
        
        latency_row = comparison_df[comparison_df['Metric'] == 'Avg Latency (ms)']
        if not latency_row.empty:
            latency_improvement = latency_row['Improvement'].iloc[0]
            latency_savings = latency_improvement.replace('-', '').replace('%', '')
        else:
            latency_savings = "N/A"
    else:
        token_savings = context_savings = latency_savings = "N/A"
    
    # 5. Generate summary text file
    summary = f"""
# RCR Router: Role-Aware Context Routing Results

## Key Achievements
- Token Savings: {token_savings}% reduction vs broadcast-all
- Context Efficiency: {context_savings}% reduction in context tokens  
- Latency Improvement: {latency_savings}% faster response times
- Multi-provider Support: Gemini, Groq, Ollama with automatic fallback

## Technical Highlights
- BERT-based semantic similarity scoring
- NP-hard knapsack optimization with polynomial fallback
- Dynamic memory versioning and conflict resolution
- Token-budgeted role-aware context routing

## Agents Performance
{agent_df.to_string(index=False) if not agent_df.empty else "No agent data available"}

## Implementation
- Based on research paper: "RCR-Router: Efficient Role-Aware Context Routing"
- Production-ready with AutoGen integration
- Comprehensive analytics and tracing
"""
    
    with open(f"{output_dir}/summary.md", "w") as f:
        f.write(summary)
    
    print(f"\n🎉 report generated in {output_dir}/")
    print("📁 Files created:")
    print("   - rcr_architecture.png (Architecture diagram)")
    print("   - token_savings.png (Performance comparison)")
    print("   - agent_performance.png (Per-agent breakdown)")
    print("   - summary.md (Copy-paste summary)")
    
    return output_dir