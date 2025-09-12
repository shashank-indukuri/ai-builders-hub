# rcr_analytics_integration.py
"""
Easy analytics integration for existing RCR Router system.
Simply import and wrap your existing manager to get comprehensive analytics.
"""

from flexible_autogen_rcr_integration import FlexibleRCRGroupChatManager
from analytics_harness import RCRAnalytics, LangSmithIntegration
from visualization_suite import RCRVisualizer, generate_ready_report
import time
from typing import Dict, List, Any

class AnalyticsEnabledRCRManager(FlexibleRCRGroupChatManager):
    """Drop-in replacement for your existing manager with full analytics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analytics = RCRAnalytics()
        self.langsmith = LangSmithIntegration()
        self.turn_start_times = {}
        self.memory_snapshots = []
        
    def initiate_chat(self, initial_message: str, max_rounds: int = 10) -> Dict[str, Any]:
        """Enhanced chat with comprehensive analytics tracking"""
        
        # Run original chat
        result = super().initiate_chat(initial_message, max_rounds)
        
        # Generate analytics after session
        self._generate_session_analytics()
        
        return result
    
    def _select_next_speaker(self, last_speaker: str = None) -> str:
        """Override to capture timing and analytics"""
        
        # Record turn start time
        selected_speaker = super()._select_next_speaker(last_speaker)
        self.turn_start_times[selected_speaker] = time.time()
        
        # Capture memory snapshot
        memory_stats = self.memory.get_memory_stats()
        self.memory_snapshots.append({
            'round': self.turn_counter,
            'total_items': memory_stats['total_items'],
            'avg_confidence': memory_stats['avg_confidence'],
            'total_tokens': memory_stats['total_tokens'],
            'by_type': dict(memory_stats['by_type'])
        })
        
        return selected_speaker
    
    def _update_coordination_state(self, agent_name: str, message_content: str):
        """Override to capture detailed metrics"""
        
        # Get agent info
        agent = self.agents.get(agent_name)
        if not agent:
            return
            
        role = agent.role
        stage = self.coordination_state.current_stage
        
        # Calculate timing
        start_time = self.turn_start_times.get(agent_name, time.time())
        end_time = time.time()
        
        # Estimate tokens (you can enhance this with actual LLM response data)
        input_tokens = len(message_content.split()) * 1.3  # Rough estimate
        output_tokens = len(message_content.split())
        
        # Get routed context info
        budget = self.role_budgets.get(role, 1000)
        routed = self.memory.query_routed(
            role=role,
            stage=stage,
            query_text=message_content[:200],  # First 200 chars for context
            budget_tokens=budget,
            use_exact_knapsack=len(self.memory.items) <= 20
        )
        
        context_tokens = sum(item.tokens for item in routed)
        
        # Get provider info (simulate based on your preferences)
        provider_preferences = {
            "planner": "gemini",
            "coder": "groq", 
            "reviewer": "ollama"
        }
        provider_used = provider_preferences.get(role, "unknown")
        
        # Get quality score
        quality_score = self.coordination_state.quality_scores.get(agent_name, 0.0)
        
        # Record analytics
        self.analytics.record_turn(
            agent_name=agent_name,
            role=role,
            round_number=self.turn_counter,
            stage=stage,
            provider_used=provider_used,
            start_time=start_time,
            end_time=end_time,
            time_to_first_token=0.5,  # Estimated
            total_latency_ms=int((end_time - start_time) * 1000),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            context_tokens=context_tokens,
            memory_items_considered=len(self.memory.items),
            memory_items_selected=len(routed),
            quality_score=quality_score
        )
        
        # Record LangSmith trace
        self.langsmith.log_agent_turn(
            agent_name=agent_name,
            role=role,
            round_number=self.turn_counter,
            input_context=f"Context with {len(routed)} items",
            output_text=message_content,
            memory_items_used=[{
                "type": item.type,
                "importance_score": getattr(item, 'importance_score', 0),
                "tokens": item.tokens,
                "confidence": item.confidence
            } for item in routed],
            token_budget=budget,
            tokens_used=context_tokens,
            provider=provider_used,
            latency_ms=int((end_time - start_time) * 1000)
        )
        
        # Call original method
        super()._update_coordination_state(agent_name, message_content)
    
    def _generate_session_analytics(self):
        """Generate comprehensive analytics after session"""
        
        print("\n" + "="*60)
        print("🔬 RCR ROUTER ANALYTICS REPORT")
        print("="*60)
        
        # 1. Generate comparison table
        comparison_df = self.analytics.generate_comparison_table()
        print("\n📊 BROADCAST ALL vs ROLE-AWARE ROUTING COMPARISON")
        print("-" * 55)
        print(comparison_df.to_string(index=False))
        
        # 2. Generate per-agent breakdown
        agent_df = self.analytics.generate_per_agent_breakdown()
        if not agent_df.empty:
            print("\n👥 PER-AGENT PERFORMANCE BREAKDOWN")
            print("-" * 40)
            print(agent_df.to_string(index=False))
        
        # 3. Budget assertions
        budget_assertions = self.langsmith.compute_budget_assertions()
        print(f"\n⚡ BUDGET EFFICIENCY ANALYSIS")
        print("-" * 35)
        print(f"Total Turns: {budget_assertions.get('total_turns', 0)}")
        print(f"Budget Violations: {budget_assertions.get('budget_violations', 0)}")
        print(f"Avg Budget Utilization: {budget_assertions.get('avg_budget_utilization', 0):.1%}")
        print(f"Avg Memory Relevance: {budget_assertions.get('avg_memory_importance', 0):.2f}")
        
        assertions_summary = budget_assertions.get('assertion_summary', {})
        print(f"\n✅ QUALITY CHECKS")
        print(f"   Budgets Respected: {assertions_summary.get('all_budgets_respected', False)}")
        print(f"   Efficient Utilization: {assertions_summary.get('efficient_utilization', False)}")
        print(f"   High Memory Relevance: {assertions_summary.get('high_memory_relevance', False)}")
        
        # 4. Generate visualizations
        print(f"\n🎨 GENERATING VISUALIZATIONS...")
        output_dir = generate_ready_report(
            self.analytics,
            comparison_df, 
            agent_df,
            "report"
        )
        
        # 5. Export traces
        self.analytics.save_traces("rcr_traces")
        self.langsmith.export_traces("langsmith_traces.json")
        
        print(f"\n🎉 ANALYTICS COMPLETE!")
        print(f"📁 Check these folders for outputs:")
        print(f"   - report/ (Visual report)")
        print(f"   - rcr_traces/ (Detailed CSV reports)")
        print(f"   - langsmith_traces.json (Trace analysis)")

# Simple usage example for your existing code
def add_analytics_to_demo():
    """Shows how to add analytics to your existing demo function"""
    
    # Import your existing components
    from flexible_autogen_rcr_integration import setup_flexible_rcr_system, create_flexible_agents
    from enhanced_rcr_router import SharedMemory, ImportanceWeights
    
    # Setup system (same as your existing code)
    manager, preferences = setup_flexible_rcr_system(
        provider_preferences = {
            "planner": "gemini",
            "coder": "groq",
            "reviewer": "ollama"
        }
    )
    
    # Enhanced memory configuration (same as existing)
    weights = ImportanceWeights(
        role_relevance=0.8,
        stage_priority=0.6, 
        recency=0.4,
        semantic_similarity=1.2,
        decision_boost=1.0,
        plan_boost=0.6
    )
    
    memory = SharedMemory(weights)
    role_budgets = {
        "planner": 2000,
        "coder": 1500,
        "reviewer": 1200
    }
    
    # Create agents (same as existing)
    agents = create_flexible_agents(manager, preferences)
    
    # Create coordination state
    from flexible_autogen_rcr_integration import AgentCoordinationState
    coordination_state = AgentCoordinationState()
    
    # 🎯 HERE'S THE ONLY CHANGE: Use analytics-enabled manager
    analytics_manager = AnalyticsEnabledRCRManager(
        agents=agents,
        memory=memory,
        role_budgets=role_budgets,
        coordination_state=coordination_state,
        llm_manager=manager
    )
    
    # Run your task (same as existing)
    task = """Create a Python function that processes CSV files with the following requirements:
1. Read multiple CSV files from a directory
2. Validate data types and handle missing values
3. Perform basic statistical analysis
4. Generate a summary report
5. Include comprehensive error handling and logging

The function should be production-ready with proper documentation."""
    
    # Execute with analytics
    result = analytics_manager.initiate_chat(task, max_rounds=8)
    
    return result

if __name__ == "__main__":
    # Run the enhanced demo
    print("🚀 Running RCR Router with Full Analytics...")
    result = add_analytics_to_demo()