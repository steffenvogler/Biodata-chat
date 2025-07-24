#!/usr/bin/env python3
"""
Advanced Reasoning + Acting (ReAct) Paradigm for Complex Research Questions

This module implements a sophisticated reasoning agent that can:
1. Decompose complex research questions into actionable steps
2. Plan multi-step research strategies
3. Execute database queries systematically
4. Synthesize findings from multiple sources
5. Reflect on results and iterate if needed
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime


class ReasoningStep(Enum):
    """Types of reasoning steps in the ReAct paradigm"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"


@dataclass
class ResearchStep:
    """A single step in the research process"""
    step_type: ReasoningStep
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchQuery:
    """A structured research query with context"""
    original_question: str
    domain: str = "general"
    complexity: str = "medium"  # low, medium, high
    required_databases: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence collected from database queries"""
    source: str
    query: str
    data: Any
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReActAgent:
    """
    Advanced Reasoning + Acting agent for complex research questions
    
    Implements the ReAct paradigm:
    - Thought: Reasoning about what to do next
    - Action: Executing a specific database query or research step
    - Observation: Analyzing the results
    - Reflection: Evaluating progress and adjusting strategy
    - Synthesis: Combining findings into coherent answers
    """
    
    def __init__(self, database_clients: Dict[str, Any], llm_backend=None, verbose: bool = False):
        self.database_clients = database_clients
        self.llm_backend = llm_backend
        self.verbose = verbose
        
        # Research state
        self.current_query: Optional[ResearchQuery] = None
        self.research_steps: List[ResearchStep] = []
        self.evidence_base: List[Evidence] = []
        self.reasoning_depth = 0
        self.max_reasoning_depth = 8
        
        # Domain knowledge for better query planning
        self.domain_expertise = {
            "taxonomy": ["bionomia", "eol", "ckan"],
            "ecology": ["eol", "ckan"],
            "collections": ["bionomia", "ckan"],
            "morphology": ["eol"],
            "behavior": ["eol"],
            "conservation": ["eol", "ckan"],
            "geographic": ["bionomia", "ckan"],
            "temporal": ["bionomia", "ckan"]
        }
        
        # Query patterns for different research types
        self.research_patterns = {
            "species_profile": [
                "taxonomic_classification",
                "morphological_description", 
                "ecological_role",
                "geographic_distribution",
                "conservation_status"
            ],
            "comparative_analysis": [
                "identify_subjects",
                "gather_comparative_data",
                "analyze_differences",
                "synthesize_patterns"
            ],
            "ecological_investigation": [
                "species_interactions",
                "habitat_requirements",
                "environmental_factors",
                "ecosystem_role"
            ],
            "historical_research": [
                "collection_records",
                "collector_information",
                "temporal_patterns",
                "geographic_changes"
            ]
        }
    
    def log_verbose(self, message: str):
        """Log verbose messages if verbose mode is enabled"""
        if self.verbose:
            print(f"[ReAct] {message}")
    
    async def analyze_question_complexity(self, question: str) -> ResearchQuery:
        """
        Analyze the complexity and requirements of a research question
        """
        self.log_verbose(f"Analyzing question complexity: {question}")
        
        # Extract domain keywords
        domain_keywords = {
            "taxonomy": ["species", "genus", "family", "order", "class", "phylum", "classification", "taxonomy"],
            "ecology": ["habitat", "ecosystem", "interaction", "predator", "prey", "symbiosis", "competition"],
            "morphology": ["anatomy", "structure", "morphology", "body", "size", "color", "shape"],
            "behavior": ["behavior", "mating", "feeding", "migration", "social", "territorial"],
            "conservation": ["endangered", "threatened", "conservation", "extinction", "protection", "status"],
            "geographic": ["distribution", "range", "location", "geography", "region", "continent"],
            "collections": ["collector", "specimen", "museum", "collection", "type specimen", "holotype"]
        }
        
        # Determine primary domain
        question_lower = question.lower()
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k]) if domain_scores else "general"
        
        # Assess complexity based on question structure
        complexity_indicators = {
            "high": ["compare", "contrast", "relationship", "evolution", "phylogeny", "multiple", "across", "between"],
            "medium": ["how", "why", "what causes", "mechanism", "process", "interaction"],
            "low": ["what is", "where", "when", "who", "which", "define"]
        }
        
        complexity = "low"
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                complexity = level
                break
        
        # Determine required databases based on domain
        required_databases = self.domain_expertise.get(primary_domain, ["bionomia", "eol", "ckan"])
        
        # Generate sub-questions for complex queries
        sub_questions = await self.decompose_question(question, primary_domain)
        
        return ResearchQuery(
            original_question=question,
            domain=primary_domain,
            complexity=complexity,
            required_databases=required_databases,
            sub_questions=sub_questions,
            context={"domain_scores": domain_scores}
        )
    
    async def decompose_question(self, question: str, domain: str) -> List[str]:
        """
        Decompose a complex question into manageable sub-questions
        """
        self.log_verbose(f"Decomposing question for domain: {domain}")
        
        # Use domain-specific patterns to generate sub-questions
        if domain in ["taxonomy", "species_profile"]:
            return [
                f"What is the taxonomic classification of the subject in: {question}?",
                f"What are the key characteristics mentioned in: {question}?",
                f"What is the distribution/habitat related to: {question}?",
                f"What ecological relationships are relevant to: {question}?"
            ]
        elif domain == "ecology":
            return [
                f"What species are involved in: {question}?",
                f"What type of ecological interaction is described in: {question}?",
                f"What environmental factors influence: {question}?",
                f"What is the broader ecosystem context of: {question}?"
            ]
        elif domain == "collections":
            return [
                f"Who are the relevant collectors for: {question}?",
                f"What specimens are related to: {question}?",
                f"What institutions hold collections related to: {question}?",
                f"What time period is relevant to: {question}?"
            ]
        else:
            # Generic decomposition
            return [
                f"What are the key entities in: {question}?",
                f"What relationships are important for: {question}?",
                f"What contextual information is needed for: {question}?"
            ]
    
    async def plan_research_strategy(self, query: ResearchQuery) -> List[str]:
        """
        Plan a multi-step research strategy based on the query
        """
        self.log_verbose(f"Planning research strategy for: {query.original_question}")
        
        strategy_steps = []
        
        # Start with understanding the core question
        strategy_steps.append("analyze_core_question")
        
        # Based on complexity, add appropriate steps
        if query.complexity == "high":
            strategy_steps.extend([
                "decompose_into_subquestions",
                "prioritize_information_needs",
                "systematic_database_search",
                "cross_reference_findings",
                "identify_knowledge_gaps",
                "synthesize_comprehensive_answer"
            ])
        elif query.complexity == "medium":
            strategy_steps.extend([
                "identify_key_information_needs",
                "targeted_database_queries",
                "analyze_relationships",
                "synthesize_findings"
            ])
        else:  # low complexity
            strategy_steps.extend([
                "direct_database_lookup",
                "verify_information",
                "format_response"
            ])
        
        return strategy_steps
    
    async def execute_thought(self, thought: str) -> ResearchStep:
        """
        Execute a reasoning/thinking step
        """
        self.log_verbose(f"Thought: {thought}")
        
        step = ResearchStep(
            step_type=ReasoningStep.THOUGHT,
            content=thought,
            metadata={"reasoning_depth": self.reasoning_depth}
        )
        
        self.research_steps.append(step)
        return step
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> ResearchStep:
        """
        Execute a research action (database query, analysis, etc.)
        """
        self.log_verbose(f"Action: {action} with parameters: {parameters}")
        
        action_result = None
        confidence = 1.0
        
        try:
            if action == "query_database":
                action_result = await self._query_database(
                    parameters.get("database", ""),
                    parameters.get("query", ""),
                    parameters.get("query_type", "search")
                )
            elif action == "analyze_relationships":
                action_result = await self._analyze_relationships(parameters.get("data", []))
            elif action == "cross_reference":
                action_result = await self._cross_reference_data(parameters.get("sources", []))
            elif action == "synthesize_information":
                action_result = await self._synthesize_information(parameters.get("evidence", []))
            else:
                action_result = f"Unknown action: {action}"
                confidence = 0.0
                
        except Exception as e:
            action_result = f"Action failed: {str(e)}"
            confidence = 0.0
        
        step = ResearchStep(
            step_type=ReasoningStep.ACTION,
            content=f"Action: {action}\nResult: {action_result}",
            confidence=confidence,
            metadata={"action": action, "parameters": parameters, "result": action_result}
        )
        
        self.research_steps.append(step)
        return step
    
    async def execute_observation(self, observation: str, data: Any = None) -> ResearchStep:
        """
        Execute an observation step - analyzing results from actions
        """
        self.log_verbose(f"Observation: {observation}")
        
        step = ResearchStep(
            step_type=ReasoningStep.OBSERVATION,
            content=observation,
            metadata={"data": data}
        )
        
        self.research_steps.append(step)
        return step
    
    async def execute_reflection(self, reflection: str) -> ResearchStep:
        """
        Execute a reflection step - evaluating progress and strategy
        """
        self.log_verbose(f"Reflection: {reflection}")
        
        step = ResearchStep(
            step_type=ReasoningStep.REFLECTION,
            content=reflection,
            metadata={"step_count": len(self.research_steps)}
        )
        
        self.research_steps.append(step)
        return step
    
    async def _query_database(self, database: str, query: str, query_type: str = "search") -> str:
        """
        Execute a database query through the appropriate MCP client
        """
        self.log_verbose(f"Querying {database} with: {query}")
        
        # This is a placeholder implementation
        # In a real implementation, this would use the actual MCP clients
        
        if database == "bionomia":
            return f"Bionomia search results for '{query}': [simulated collector and specimen data]"
        elif database == "eol":
            return f"EOL search results for '{query}': [simulated species and trait data]"
        elif database == "ckan":
            return f"CKAN search results for '{query}': [simulated dataset and collection data]"
        else:
            return f"Unknown database: {database}"
    
    async def _analyze_relationships(self, data: List[Any]) -> str:
        """
        Analyze relationships in the collected data
        """
        if not data:
            return "No data to analyze"
        
        # Placeholder relationship analysis
        return f"Analyzed {len(data)} data points: [simulated relationship patterns]"
    
    async def _cross_reference_data(self, sources: List[str]) -> str:
        """
        Cross-reference data from multiple sources
        """
        return f"Cross-referenced data from {len(sources)} sources: [simulated cross-references]"
    
    async def _synthesize_information(self, evidence: List[Evidence]) -> str:
        """
        Synthesize information from collected evidence
        """
        if not evidence:
            return "No evidence to synthesize"
        
        return f"Synthesized information from {len(evidence)} pieces of evidence: [simulated synthesis]"
    
    async def should_continue_reasoning(self) -> bool:
        """
        Determine if the reasoning process should continue
        """
        if self.reasoning_depth >= self.max_reasoning_depth:
            return False
        
        # Check if we have sufficient evidence
        if len(self.evidence_base) >= 3 and self.reasoning_depth >= 3:
            return False
        
        # Check if recent steps are productive
        recent_steps = self.research_steps[-3:] if len(self.research_steps) >= 3 else self.research_steps
        if recent_steps:
            avg_confidence = sum(step.confidence for step in recent_steps) / len(recent_steps)
            if avg_confidence < 0.3:  # Low confidence suggests we should stop
                return False
        
        return True
    
    async def research_question(self, question: str) -> Dict[str, Any]:
        """
        Main method to research a complex question using ReAct paradigm
        """
        self.log_verbose(f"Starting ReAct research for: {question}")
        
        # Reset state for new question
        self.current_query = None
        self.research_steps = []
        self.evidence_base = []
        self.reasoning_depth = 0
        
        try:
            # Step 1: Analyze the question
            self.current_query = await self.analyze_question_complexity(question)
            
            await self.execute_thought(
                f"I need to research: '{question}'. "
                f"This appears to be a {self.current_query.complexity} complexity question "
                f"in the {self.current_query.domain} domain. "
                f"I should use databases: {', '.join(self.current_query.required_databases)}"
            )
            
            # Step 2: Plan research strategy  
            strategy = await self.plan_research_strategy(self.current_query)
            
            await self.execute_thought(
                f"My research strategy will involve: {', '.join(strategy)}"
            )
            
            # Step 3: Execute ReAct reasoning loop
            while await self.should_continue_reasoning():
                self.reasoning_depth += 1
                
                # Thought: What should I do next?
                if self.reasoning_depth == 1:
                    thought = f"Let me start by searching the most relevant database for core information about this question."
                elif self.reasoning_depth <= 3:
                    thought = f"I need to gather more specific information to build a comprehensive understanding."
                else:
                    thought = f"Let me analyze what I've found so far and determine if I need additional information."
                
                await self.execute_thought(thought)
                
                # Action: Execute a research action
                if self.reasoning_depth <= len(self.current_query.required_databases):
                    # Query each required database
                    db_idx = self.reasoning_depth - 1
                    database = self.current_query.required_databases[db_idx]
                    
                    action_step = await self.execute_action(
                        "query_database",
                        {
                            "database": database,
                            "query": question,
                            "query_type": "comprehensive_search"
                        }
                    )
                    
                    # Add evidence
                    evidence = Evidence(
                        source=database,
                        query=question,
                        data=action_step.metadata.get("result", ""),
                        confidence=action_step.confidence
                    )
                    self.evidence_base.append(evidence)
                    
                else:
                    # Analyze and synthesize
                    await self.execute_action(
                        "synthesize_information",
                        {"evidence": self.evidence_base}
                    )
                
                # Observation: What did I learn?
                if self.evidence_base:
                    latest_evidence = self.evidence_base[-1]
                    observation = (
                        f"From {latest_evidence.source}, I found information about the question. "
                        f"Confidence level: {latest_evidence.confidence:.2f}"
                    )
                else:
                    observation = "I need to gather more information to properly answer this question."
                
                await self.execute_observation(observation)
                
                # Reflection: How am I doing?
                if self.reasoning_depth % 2 == 0:  # Reflect every other step
                    reflection = (
                        f"After {self.reasoning_depth} reasoning steps, I have gathered "
                        f"{len(self.evidence_base)} pieces of evidence. "
                        f"{'I should continue gathering information.' if len(self.evidence_base) < 3 else 'I have sufficient information to synthesize an answer.'}"
                    )
                    await self.execute_reflection(reflection)
            
            # Final synthesis
            final_synthesis = await self.execute_action(
                "synthesize_information",
                {"evidence": self.evidence_base}
            )
            
            # Compile final answer
            final_answer = await self._compile_final_answer()
            
            return {
                "question": question,
                "answer": final_answer,
                "reasoning_steps": len(self.research_steps),
                "evidence_count": len(self.evidence_base),
                "complexity": self.current_query.complexity,
                "domain": self.current_query.domain,
                "databases_used": self.current_query.required_databases,
                "research_trace": [
                    {
                        "step": i+1,
                        "type": step.step_type.value,
                        "content": step.content[:200] + "..." if len(step.content) > 200 else step.content,
                        "confidence": step.confidence
                    }
                    for i, step in enumerate(self.research_steps)
                ]
            }
            
        except Exception as e:
            return {
                "question": question,
                "error": f"Research failed: {str(e)}",
                "reasoning_steps": len(self.research_steps),
                "evidence_count": len(self.evidence_base)
            }
    
    async def _compile_final_answer(self) -> str:
        """
        Compile the final answer from all reasoning steps and evidence
        """
        if not self.evidence_base:
            return "I was unable to find sufficient information to answer this question."
        
        # Synthesize evidence into a coherent answer
        answer_parts = []
        
        answer_parts.append(f"Based on my research across {len(self.current_query.required_databases)} scientific databases, here's what I found:")
        
        # Add evidence summaries
        for i, evidence in enumerate(self.evidence_base, 1):
            answer_parts.append(f"\n{i}. From {evidence.source}: {evidence.data}")
        
        # Add reasoning summary
        answer_parts.append(f"\nThrough {self.reasoning_depth} reasoning steps, I analyzed this {self.current_query.complexity} complexity question in the {self.current_query.domain} domain.")
        
        # Add synthesis
        if len(self.evidence_base) > 1:
            answer_parts.append("\nSynthesizing across sources, the key findings suggest a comprehensive understanding of the research question.")
        
        return " ".join(answer_parts)


# Export the main class
__all__ = ["ReActAgent", "ResearchQuery", "Evidence", "ReasoningStep"]
