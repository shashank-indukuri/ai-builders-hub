# flexible_autogen_rcr_integration.py
import autogen
from enhanced_rcr_router import SharedMemory, extract_structured, pack_context, JSON_SCHEMA_INSTRUCTION, evaluate_answer_quality, ImportanceWeights
from flexible_llm_provider import LLMProviderManager, setup_simple_manager, LLMResponse, ProviderType, LLMConfig
from typing import List, Dict, Any, Optional, Callable
import logging
import time
import os
from dataclasses import dataclass, field

@dataclass 
class FlexibleAgentConfig:
    """Configuration for agents with flexible LLM providers"""
    name: str
    role: str  # planner, coder, reviewer, etc.
    system_message: str
    preferred_provider: Optional[str] = None  # e.g., "gemini-flash", "groq-llama3"
    fallback_providers: List[str] = field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 4000

@dataclass 
class AgentCoordinationState:
    """Track agent dependencies and coordination state"""
    pending_agents: List[str] = field(default_factory=list)
    completed_agents: List[str] = field(default_factory=list) 
    agent_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    current_stage: str = "plan"
    stage_completion: Dict[str, bool] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

class FlexibleAssistantAgent:
    """AutoGen-compatible agent that uses flexible LLM providers"""
    
    def __init__(self, config: FlexibleAgentConfig, llm_manager: LLMProviderManager):
        self.name = config.name
        self.role = config.role
        self.system_message = config.system_message
        self.preferred_provider = config.preferred_provider
        self.fallback_providers = config.fallback_providers
        self.llm_manager = llm_manager
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.conversation_history = []
        
        # For AutoGen compatibility
        self.human_input_mode = "NEVER"
        self.max_consecutive_auto_reply = 1
        
    def generate_reply(self, messages: List[Dict], sender=None, config=None) -> str:
        """Generate reply using flexible LLM providers"""
        
        # Prepare messages for LLM
        llm_messages = []
        
        # Add system message
        if self.system_message:
            llm_messages.append({"role": "system", "content": self.system_message})
        
        # Add conversation messages
        for msg in messages:
            if isinstance(msg, dict):
                llm_messages.append(msg)
            else:
                # Handle AutoGen message format
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))
                llm_messages.append({"role": role, "content": content})
        
        # Generate response with fallback
        providers_to_try = []
        if self.preferred_provider:
            providers_to_try.append(self.preferred_provider)
        providers_to_try.extend(self.fallback_providers)
        
        response = self.llm_manager.generate_response(
            llm_messages, 
            preferred_provider=self.preferred_provider
        )
        
        if response.error:
            logging.error(f"Agent {self.name} failed to generate response: {response.error}")
            return f"Error: Failed to generate response. {response.error}"
        
        # Log usage info
        if response.usage:
            logging.info(f"Agent {self.name} used {response.provider}: "
                        f"{response.usage.get('total_tokens', 'N/A')} tokens, "
                        f"{response.latency_ms}ms")
        
        return response.content

class FlexibleRCRGroupChatManager:
    """Enhanced GroupChatManager with flexible providers and deep RCR integration"""
    
    def __init__(self, agents: List[FlexibleAssistantAgent], memory: SharedMemory, 
                 role_budgets: Dict[str, int], coordination_state: AgentCoordinationState,
                 llm_manager: LLMProviderManager):
        self.agents = {agent.name: agent for agent in agents}
        self.memory = memory
        self.role_budgets = role_budgets
        self.coordination_state = coordination_state
        self.llm_manager = llm_manager
        self.turn_counter = 0
        self.conversation_quality_scores = []
        self.conversation_history = []
        
    def _inject_rcr_context(self, agent_name: str, messages: List[Dict]) -> List[Dict]:
        """Inject role-scoped context before agent processes message"""
        agent = self.agents.get(agent_name)
        if not agent:
            return messages
            
        role = agent.role
        stage = self.coordination_state.current_stage
        
        # Get last user message for query context
        last_msg = messages[-1]["content"] if messages else ""
        
        # Route context based on role, stage, and query
        budget = self.role_budgets.get(role, 1000)
        routed = self.memory.query_routed(
            role=role,
            stage=stage, 
            query_text=last_msg,
            budget_tokens=budget,
            use_exact_knapsack=len(self.memory.items) <= 20
        )
        
        # Create context message
        ctx_content = pack_context(routed, include_metadata=True)
        system_msg = {
            "role": "system", 
            "content": f"{ctx_content}\n\n{JSON_SCHEMA_INSTRUCTION}"
        }
        
        # Insert system message before the last user message
        enhanced_messages = messages[:-1] + [system_msg] + [messages[-1]]
        return enhanced_messages

    def _update_coordination_state(self, agent_name: str, message_content: str):
        """Update coordination state based on agent output"""
        agent = self.agents.get(agent_name)
        if not agent:
            return
            
        role = agent.role
        
        # Mark agent as completed for current stage
        if agent_name not in self.coordination_state.completed_agents:
            self.coordination_state.completed_agents.append(agent_name)
            
        # Extract structured items and update memory
        items = extract_structured(
            message_content,
            agent_name,
            self.turn_counter,
            [role],
            [self.coordination_state.current_stage]
        )
        self.memory.add(items)
        
        # Check if we should advance stage
        self._check_stage_advancement()
        
        # Evaluate answer quality using flexible providers
        if hasattr(self, '_last_user_query'):
            quality_score, justification = evaluate_answer_quality(
                self._last_user_query, message_content
            )
            self.coordination_state.quality_scores[agent_name] = quality_score
            self.conversation_quality_scores.append(quality_score)
            logging.info(f"Quality score for {agent_name}: {quality_score:.2f}")

    def _check_stage_advancement(self):
        """Determine if we should advance to next stage"""
        stage_requirements = {
            "plan": ["planner"],
            "execute": ["coder"], 
            "review": ["reviewer"]
        }
        
        current_stage = self.coordination_state.current_stage
        required_roles = stage_requirements.get(current_stage, [])
        
        # Check if all required roles have completed
        completed_roles = [self.agents[name].role for name in self.coordination_state.completed_agents if name in self.agents]
        
        if all(role in completed_roles for role in required_roles):
            # Advance to next stage
            stage_order = ["plan", "execute", "review"]
            current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
            
            if current_idx < len(stage_order) - 1:
                next_stage = stage_order[current_idx + 1]
                self.coordination_state.current_stage = next_stage
                self.coordination_state.completed_agents.clear()
                logging.info(f"Advanced to stage: {next_stage}")

    def _select_next_speaker(self, last_speaker: str = None) -> str:
        """Select next speaker based on coordination state"""
        stage = self.coordination_state.current_stage
        
        # Stage-based selection
        if stage == "plan":
            planners = [name for name, agent in self.agents.items() if agent.role == "planner"]
            if planners:
                return planners[0]
                
        elif stage == "execute":
            coders = [name for name, agent in self.agents.items() if agent.role == "coder"]
            if coders:
                return coders[0]
                
        elif stage == "review":
            reviewers = [name for name, agent in self.agents.items() if agent.role == "reviewer"]
            if reviewers:
                return reviewers[0]
        
        # Fallback to round-robin
        agent_names = list(self.agents.keys())
        if last_speaker and last_speaker in agent_names:
            current_idx = agent_names.index(last_speaker)
            next_idx = (current_idx + 1) % len(agent_names)
            return agent_names[next_idx]
        
        return agent_names[0] if agent_names else ""

    def initiate_chat(self, initial_message: str, max_rounds: int = 10) -> Dict[str, Any]:
        """Run the multi-agent conversation"""
        self._last_user_query = initial_message
        self.conversation_history = [{"role": "user", "content": initial_message}]
        
        last_speaker = None
        
        for round_num in range(max_rounds):
            self.turn_counter = round_num + 1
            
            # Select next speaker
            current_speaker_name = self._select_next_speaker(last_speaker)
            if not current_speaker_name:
                break
                
            current_agent = self.agents[current_speaker_name]
            
            # Prepare messages with RCR context
            messages = self._inject_rcr_context(current_speaker_name, self.conversation_history.copy())
            
            # Generate response
            response = current_agent.generate_reply(messages)
            
            # Add response to conversation
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
                "name": current_speaker_name
            })
            
            # Update coordination state
            self._update_coordination_state(current_speaker_name, response)
            
            # Check for termination conditions
            if self._should_terminate(response):
                break
                
            last_speaker = current_speaker_name
            
        return {
            "conversation": self.conversation_history,
            "turns": self.turn_counter,
            "final_stage": self.coordination_state.current_stage,
            "quality_scores": self.coordination_state.quality_scores
        }

    def _should_terminate(self, message: str) -> bool:
        """Check if conversation should terminate"""
        termination_phrases = [
            "TERMINATE",
            "task completed",
            "conversation complete",
            "no further action needed"
        ]
        
        message_lower = message.lower()
        return any(phrase in message_lower for phrase in termination_phrases)

def create_flexible_agents(
    llm_manager: LLMProviderManager,
    provider_preferences: Dict[str, str] = None
) -> List[FlexibleAssistantAgent]:
    """Create agents with flexible LLM provider support"""
    
    if provider_preferences is None:
        provider_preferences = {
            "planner": "gemini-flash",  # Fast for planning
            "coder": "groq-llama3",     # Good for code
            "reviewer": "gemini-pro"    # Thorough for review
        }
    
    agent_configs = [
        FlexibleAgentConfig(
            name="planner",
            role="planner", 
            system_message="""You are a strategic planner. Analyze tasks and create structured plans.
Always respond with JSON containing decisions, plans, and identified entities.
Focus on breaking down complex tasks into actionable steps.""",
            preferred_provider=provider_preferences.get("planner"),
            fallback_providers=["groq-llama3", "gemini-flash", "ollama-llama3"]
        ),
        
        FlexibleAgentConfig(
            name="coder",
            role="coder",
            system_message="""You are a code executor. Implement plans and write functional code.
Always respond with JSON containing tool_traces, facts about implementation, and any decisions made.
Focus on practical implementation and testing.""",
            preferred_provider=provider_preferences.get("coder"),
            fallback_providers=["groq-llama3", "ollama-codellama", "gemini-flash"]
        ),
        
        FlexibleAgentConfig(
            name="reviewer", 
            role="reviewer",
            system_message="""You are a quality reviewer. Evaluate outputs and provide feedback.
Always respond with JSON containing decisions about quality, entities that need attention, and facts about the review.
Focus on accuracy, completeness, and improvement suggestions.""",
            preferred_provider=provider_preferences.get("reviewer"),
            fallback_providers=["gemini-pro", "groq-llama3", "gemini-flash"]
        )
    ]
    
    agents = []
    for config in agent_configs:
        agents.append(FlexibleAssistantAgent(config, llm_manager))
    
    return agents

def run_flexible_multi_agent_session(
    task: str,
    llm_manager: LLMProviderManager,
    max_rounds: int = 10,
    memory: Optional[SharedMemory] = None,
    weights: Optional[ImportanceWeights] = None,
    role_budgets: Optional[Dict[str, int]] = None,
    provider_preferences: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Run a complete multi-agent session with flexible providers"""
    
    start_time = time.time()
    
    # Initialize components
    if memory is None:
        memory = SharedMemory(weights or ImportanceWeights())
    
    if role_budgets is None:
        role_budgets = {
            "planner": 2000,
            "coder": 1500, 
            "reviewer": 1200
        }
    
    # Create agents
    agents = create_flexible_agents(llm_manager, provider_preferences)
    
    # Create coordination state
    coordination_state = AgentCoordinationState()
    
    # Create manager
    manager = FlexibleRCRGroupChatManager(
        agents=agents,
        memory=memory,
        role_budgets=role_budgets,
        coordination_state=coordination_state,
        llm_manager=llm_manager
    )
    
    # Run conversation
    try:
        result = manager.initiate_chat(task, max_rounds)
        success = True
        error = None
    except Exception as e:
        logging.error(f"Session failed: {e}")
        result = {"conversation": [], "turns": 0}
        success = False
        error = str(e)
    
    end_time = time.time()
    
    # Collect session statistics
    stats = {
        "success": success,
        "error": error,
        "duration_seconds": end_time - start_time,
        "total_rounds": result.get("turns", 0),
        "memory_stats": memory.get_memory_stats(),
        "coordination_state": coordination_state,
        "average_quality": sum(manager.conversation_quality_scores) / len(manager.conversation_quality_scores) if manager.conversation_quality_scores else 0.0,
        "quality_scores_by_agent": coordination_state.quality_scores,
        "conversation": result.get("conversation", []),
        "provider_usage": _analyze_provider_usage(llm_manager, agents)
    }
    
    logging.info(f"Session completed: {stats['total_rounds']} rounds, avg quality: {stats['average_quality']:.2f}")
    
    return stats

def _analyze_provider_usage(llm_manager: LLMProviderManager, agents: List[FlexibleAssistantAgent]) -> Dict[str, Any]:
    """Analyze which providers were used during the session"""
    usage_stats = {
        "providers_attempted": [],
        "providers_successful": [],
        "agent_provider_preferences": {}
    }
    
    for agent in agents:
        usage_stats["agent_provider_preferences"][agent.name] = {
            "preferred": agent.preferred_provider,
            "fallbacks": agent.fallback_providers
        }
    
    return usage_stats

def setup_flexible_rcr_system(
    openai_key: str = None,
    gemini_key: str = None, 
    groq_key: str = None,
    anthropic_key: str = None,
    ollama_url: str = "http://localhost:11434",
    provider_preferences: Dict[str, str] = None
) -> tuple[LLMProviderManager, Dict[str, str]]:
    """Setup the complete flexible RCR system"""
    
    # Get API keys from environment if not provided
    openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY") 
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    
    # Setup LLM manager
    # manager = setup_default_manager(
    manager = setup_simple_manager(
        # openai_key=openai_key,
        gemini_key=gemini_key,
        groq_key=groq_key,
        # anthropic_key=anthropic_key,
        ollama_url=ollama_url
    )
    
    # Test available providers
    available = manager.get_available_providers()
    logging.info(f"Available providers: {available}")
    
    # Set default preferences based on availability
    default_preferences = {}
    if "gemini" in available:
        default_preferences.update({
            "planner": "gemini-2.0-flash",
            "reviewer": "gemini-2.0-flash"
        })
    elif "groq" in available:
        default_preferences.update({
            "planner": "meta-llama/llama-4-scout-17b-16e-instruct",
            "reviewer": "meta-llama/llama-4-scout-17b-16e-instruct"
        })
    elif "ollama" in available:
        default_preferences.update({
            "planner": "qwen2.5vl:latest",
            "reviewer": "qwen2.5vl:latest"
        })
    
    if "groq" in available:
        default_preferences["coder"] = "meta-llama/llama-4-scout-17b-16e-instruct"
    elif "ollama" in available:
        default_preferences["coder"] = "qwen2.5vl:latest"
    elif available:
        default_preferences["coder"] = available[0]
    
    # Use provided preferences or defaults
    final_preferences = provider_preferences or default_preferences
    
    return manager, final_preferences

def demo_flexible_rcr():
    """Demonstrate the flexible RCR system with multiple providers"""
    
    # Setup system (add your API keys)
    manager, preferences = setup_flexible_rcr_system(
        # Uncomment and add your keys:
        # openai_key="your-openai-key",
        # gemini_key="your-gemini-key", 
        # groq_key="your-groq-key",
        # ollama_url="http://localhost:11434",
        # anthropic_key="your-anthropic-key"
        provider_preferences = {
                "planner": "gemini" ,#gemini-2.0-flash",    # Fast planning
                "coder": "groq", #"meta-llama/llama-4-scout-17b-16e-instruct",       # Good at code
                "reviewer": "ollama" #"qwen2.5vl:latest"      # Thorough review
                }
    )
    
    print("\n=== Flexible RCR System Demo ===")
    print(f"Available providers: {manager.get_available_providers()}")
    print(f"Provider preferences: {preferences}")
    
    # Test all providers
    from flexible_llm_provider import test_providers
    test_providers(manager)
    
    # Enhanced memory configuration
    weights = ImportanceWeights(
        role_relevance=0.8,
        stage_priority=0.6, 
        recency=0.4,
        semantic_similarity=1.2,
        decision_boost=1.0,
        plan_boost=0.6
    )
    
    memory = SharedMemory(weights)
    
    # Role budgets optimized for different provider costs
    role_budgets = {
        "planner": 2000,  # More context for planning
        "coder": 1500,    # Substantial context for coding
        "reviewer": 1200  # Focused context for review
    }
    
    # Run test session
    task = """Create a Python function that processes CSV files with the following requirements:
1. Read multiple CSV files from a directory
2. Validate data types and handle missing values
3. Perform basic statistical analysis
4. Generate a summary report
5. Include comprehensive error handling and logging

The function should be production-ready with proper documentation."""
    
    results = run_flexible_multi_agent_session(
        task=task,
        llm_manager=manager,
        max_rounds=8,
        memory=memory,
        weights=weights,
        role_budgets=role_budgets,
        provider_preferences=preferences
    )
    
    # Display results
    print(f"\n=== Session Results ===")
    print(f"Success: {results['success']}")
    if results['error']:
        print(f"Error: {results['error']}")
    print(f"Duration: {results['duration_seconds']:.1f} seconds")
    print(f"Rounds: {results['total_rounds']}")
    print(f"Average Quality: {results['average_quality']:.2f}")
    
    print(f"\n=== Memory Statistics ===")
    stats = results['memory_stats']
    print(f"Total Items: {stats['total_items']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Total Tokens: {stats['total_tokens']}")
    
    print(f"\n=== Quality Scores by Agent ===")
    for agent, score in results['quality_scores_by_agent'].items():
        print(f"{agent}: {score:.2f}")
    
    print(f"\n=== Memory Distribution ===")
    for item_type, count in stats['by_type'].items():
        print(f"{item_type}: {count}")
    
    return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demo
    demo_flexible_rcr()