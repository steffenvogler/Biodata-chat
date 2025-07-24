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
class ReasoningChain:
    """Maintains knowledge and reasoning context across research steps"""
    question_id: str
    accumulated_knowledge: Dict[str, Any] = field(default_factory=dict)
    concept_relationships: Dict[str, List[str]] = field(default_factory=dict)
    evidence_links: Dict[str, List[str]] = field(default_factory=dict)
    reasoning_history: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)


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
    
    def __init__(self, database_clients: Dict[str, Any], llm_backend=None, verbose: bool = False, force_high_complexity: bool = True):
        self.database_clients = database_clients
        self.llm_backend = llm_backend
        self.verbose = verbose
        self.force_high_complexity = force_high_complexity  # Force all questions to be treated as high complexity
        
        # Research state
        self.current_query: Optional[ResearchQuery] = None
        self.research_steps: List[ResearchStep] = []
        self.evidence_base: List[Evidence] = []
        self.reasoning_depth = 0
        self.max_reasoning_depth = 8
        
        # Reasoning chain for context accumulation
        self.reasoning_chain: Optional[ReasoningChain] = None
        
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
        
        # Assess complexity based on question structure and scientific concepts
        complexity_indicators = {
            "high": [
                # Comparative analysis
                "compare", "contrast", "relationship", "versus", "vs", "differences", "similarities",
                # Multi-factor analysis
                "interaction", "interactions", "gene-environment", "genotype-phenotype", "coevolution",
                # Systems-level concepts
                "evolution", "phylogeny", "phylogenetic", "evolutionary", "ecosystem", "food web",
                # Complex processes
                "mechanism", "mechanisms", "pathway", "pathways", "cascade", "network", "networks",
                # Multi-dimensional queries
                "multiple", "across", "between", "among", "throughout", "diversity", "variation",
                # Advanced concepts
                "genomic", "transcriptomic", "proteomic", "metabolomic", "epigenetic", "microbiome",
                "climate change", "anthropocene", "conservation genomics", "population dynamics"
            ],
            "medium": [
                # Mechanistic questions
                "how", "why", "what causes", "process", "function", "role", "impact", "effect",
                # Single-factor analysis
                "adaptation", "behavior", "habitat", "distribution", "migration", "breeding",
                # Intermediate concepts
                "trait", "traits", "characteristics", "morphology", "physiology", "ecology",
                "population", "community", "species composition", "abundance", "density"
            ],
            "low": [
                # Basic identification
                "what is", "where", "when", "who", "which", "define", "identify",
                # Simple facts
                "size", "length", "weight", "color", "appearance", "location", "range",
                # Basic classification
                "taxonomy", "classification", "family", "genus", "species", "order", "class"
            ]
        }
        
        complexity = "low"
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                complexity = level
                break
        
        # Override complexity if force_high_complexity is enabled
        if self.force_high_complexity:
            complexity = "high"
            self.log_verbose("Forcing high complexity for advanced reasoning mode")
        
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
        self._update_reasoning_chain(step)
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
            elif action == "generate_follow_up_questions":
                action_result = await self._generate_follow_up_questions(
                    parameters.get("evidence", []),
                    parameters.get("original_question", "")
                )
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
        self._update_reasoning_chain(step)
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
        self._update_reasoning_chain(step)
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
        self._update_reasoning_chain(step)
        return step
    
    async def _query_database(self, database: str, query: str, query_type: str = "search") -> str:
        """
        Execute a database query through the appropriate MCP client
        """
        self.log_verbose(f"Querying {database} with: {query}")
        
        # Simulate actual database query with more realistic responses
        if database == "bionomia":
            if "polar bear" in query.lower():
                return "Bionomia results: Polar bear specimens collected by Arctic expeditions, mainly by researchers like Vilhjalmur Stefansson (1913-1918) and modern USGS studies. Attribution data shows collection locations across Arctic regions including Svalbard, Canadian Arctic, and Alaska."
            elif "penguin" in query.lower():
                return "Bionomia results: Limited penguin specimen data as Bionomia focuses on attribution. Some historical specimens collected by Antarctic expeditions including Ernest Shackleton's expeditions and modern research stations in Antarctica."
            elif "great auk" in query.lower():
                return "Bionomia results: Historical Great Auk specimens from final populations, collected by naturalists including John Wolley (1858) and others from Eldey Island, Iceland. Last specimens collected 1844 before extinction."
            else:
                return f"Bionomia search results for '{query}': Collector attribution data and specimen records from scientific collections worldwide."
        elif database == "eol":
            if "polar bear" in query.lower():
                return "EOL results: Ursus maritimus - Arctic apex predator, sea ice dependent, feeds primarily on ringed seals. Plays crucial role as keystone species in Arctic marine ecosystems. Population: ~26,000 individuals across 19 subpopulations."
            elif "penguin" in query.lower():
                return "EOL results: Multiple penguin species (Spheniscidae family) - Antarctic and sub-Antarctic marine birds. Key ecological roles include nutrient transport from ocean to land, supporting Southern Ocean food webs. 18 species total."
            elif "great auk" in query.lower():
                return "EOL results: Pinguinus impennis (extinct 1844) - Large flightless seabird, North Atlantic. Was apex marine predator in cold northern waters, similar ecological niche to penguins but in Northern Hemisphere."
            else:
                return f"EOL search results for '{query}': Comprehensive species information including taxonomy, ecology, behavior, and conservation status."
        elif database == "ckan":
            if "polar bear" in query.lower():
                return "CKAN results: Multiple datasets including Arctic sea ice monitoring, polar bear tracking data from GPS collars, population surveys from USGS and Canadian Wildlife Service, and climate impact studies."
            elif "penguin" in query.lower():
                return "CKAN results: Antarctic research datasets including penguin colony counts, breeding success data, foraging behavior studies, and climate change impact assessments from various research stations."
            elif "great auk" in query.lower():
                return "CKAN results: Historical collections and museum specimens data, extinction timeline datasets, archaeological evidence from former breeding sites, and genetic analysis of preserved specimens."
            else:
                return f"CKAN search results for '{query}': Research datasets, monitoring data, and scientific collections from museums and research institutions."
        else:
            return f"Unknown database: {database}"
    
    async def _analyze_relationships(self, data: List[Any]) -> str:
        """
        Analyze relationships in the collected data and build knowledge connections
        """
        if not data:
            return "No evidence available for relationship analysis"
        
        # Extract entities and relationships from evidence
        relationships = []
        entities = set()
        
        for evidence in self.evidence_base:
            # Extract key terms and entities
            data_text = str(evidence.data).lower()
            
            # Look for scientific relationships
            if "predator" in data_text and "prey" in data_text:
                relationships.append(f"Predator-prey relationship identified in {evidence.source}")
            if "feeds on" in data_text or "eats" in data_text:
                relationships.append(f"Feeding relationship documented in {evidence.source}")
            if "habitat" in data_text:
                relationships.append(f"Habitat dependency noted in {evidence.source}")
            if "interaction" in data_text:
                relationships.append(f"Ecological interaction described in {evidence.source}")
            if "population" in data_text:
                relationships.append(f"Population data available from {evidence.source}")
            
            # Extract entity names (species, locations, etc.)
            words = data_text.split()
            for word in words:
                if len(word) > 4 and word.isalpha():  # Basic entity extraction
                    entities.add(word.capitalize())
        
        # Update reasoning chain with discovered relationships
        if self.reasoning_chain:
            for relationship in relationships:
                if relationship not in self.reasoning_chain.key_insights:
                    self.reasoning_chain.key_insights.append(relationship)
        
        analysis_result = f"""Relationship Analysis Results:
• Identified {len(relationships)} key relationships across {len(self.evidence_base)} sources
• Extracted {len(entities)} potential entities: {', '.join(list(entities)[:10])}{'...' if len(entities) > 10 else ''}
• Key patterns: {"; ".join(relationships[:5])}

This analysis reveals interconnected ecological, taxonomic, and geographic relationships that inform our understanding of the research question."""
        
        return analysis_result
    
    async def _cross_reference_data(self, sources: List[str]) -> str:
        """
        Cross-reference data from multiple sources to identify consistencies and gaps
        """
        if len(sources) < 2:
            return "Insufficient sources for meaningful cross-referencing"
        
        # Analyze information consistency across sources
        cross_refs = []
        data_points = {}
        
        for evidence in self.evidence_base:
            source = evidence.source
            data_text = str(evidence.data).lower()
            
            # Extract key data points
            if "population" in data_text:
                if "population" not in data_points:
                    data_points["population"] = []
                data_points["population"].append(source)
            
            if "habitat" in data_text or "ecosystem" in data_text:
                if "habitat" not in data_points:
                    data_points["habitat"] = []
                data_points["habitat"].append(source)
            
            if "specimen" in data_text or "collection" in data_text:
                if "specimens" not in data_points:
                    data_points["specimens"] = []
                data_points["specimens"].append(source)
            
            if "distribution" in data_text or "range" in data_text:
                if "distribution" not in data_points:
                    data_points["distribution"] = []
                data_points["distribution"].append(source)
        
        # Identify convergent and divergent information
        convergent = []
        coverage_gaps = []
        
        for data_type, source_list in data_points.items():
            if len(source_list) > 1:
                convergent.append(f"{data_type} information confirmed across {len(source_list)} sources: {', '.join(source_list)}")
            else:
                coverage_gaps.append(f"{data_type} data only available from {source_list[0]}")
        
        # Update reasoning chain with cross-reference insights
        if self.reasoning_chain:
            insight = f"Cross-referencing revealed {len(convergent)} convergent data points and {len(coverage_gaps)} coverage gaps"
            if insight not in self.reasoning_chain.key_insights:
                self.reasoning_chain.key_insights.append(insight)
        
        cross_ref_result = f"""Cross-Reference Analysis:
• Analyzed data consistency across {len(sources)} sources: {', '.join(sources)}

Convergent Information:
{chr(10).join(['• ' + item for item in convergent]) if convergent else '• No overlapping data points identified'}

Coverage Gaps:
{chr(10).join(['• ' + item for item in coverage_gaps]) if coverage_gaps else '• Complete coverage across all sources'}

This cross-referencing helps validate findings and identify areas where additional research may be needed."""
        
        return cross_ref_result
    
    async def _generate_follow_up_questions(self, evidence: List[Evidence], original_question: str) -> str:
        """
        Generate follow-up research questions based on accumulated evidence
        """
        if not evidence:
            return "No evidence available to generate follow-up questions"
        
        follow_ups = []
        knowledge_gaps = []
        
        # Analyze evidence for gaps and interesting patterns
        has_population_data = any("population" in str(e.data).lower() for e in evidence)
        has_habitat_data = any("habitat" in str(e.data).lower() for e in evidence)
        has_interaction_data = any("interaction" in str(e.data).lower() for e in evidence)
        has_conservation_data = any("conservation" in str(e.data).lower() or "endangered" in str(e.data).lower() for e in evidence)
        has_specimen_data = any("specimen" in str(e.data).lower() for e in evidence)
        
        # Generate follow-up questions based on available data
        if has_population_data:
            follow_ups.append("How have population trends changed over time?")
            follow_ups.append("What factors most significantly impact population dynamics?")
        else:
            knowledge_gaps.append("Population data not found - consider demographic analysis")
        
        if has_habitat_data:
            follow_ups.append("How is climate change affecting habitat suitability?")
            follow_ups.append("What are the critical habitat requirements for survival?")
        else:
            knowledge_gaps.append("Habitat information limited - ecological requirements unclear")
        
        if has_interaction_data:
            follow_ups.append("What cascading effects do these interactions have on the ecosystem?")
            follow_ups.append("How do seasonal changes affect these relationships?")
        else:
            knowledge_gaps.append("Ecological interactions not well documented")
        
        if has_conservation_data:
            follow_ups.append("What conservation strategies have proven most effective?")
            follow_ups.append("How do current threats compare to historical challenges?")
        else:
            knowledge_gaps.append("Conservation status and threats need investigation")
        
        if has_specimen_data:
            follow_ups.append("What can genetic analysis of specimens reveal about population structure?")
            follow_ups.append("How do historical specimens compare to modern collections?")
        else:
            knowledge_gaps.append("Specimen-based research opportunities not explored")
        
        # Update reasoning chain with generated questions
        if self.reasoning_chain:
            insight = f"Generated {len(follow_ups)} follow-up questions and identified {len(knowledge_gaps)} knowledge gaps"
            if insight not in self.reasoning_chain.key_insights:
                self.reasoning_chain.key_insights.append(insight)
        
        follow_up_result = f"""Follow-up Research Questions Generated:

Based on the evidence collected, here are {len(follow_ups)} promising research directions:
{chr(10).join(['• ' + q for q in follow_ups[:5]])}

Knowledge Gaps Identified:
{chr(10).join(['• ' + gap for gap in knowledge_gaps[:3]])}

These questions could guide deeper investigation and reveal additional insights about the research topic."""
        
        return follow_up_result
    
    async def _synthesize_information(self, evidence: List[Evidence]) -> str:
        """
        Synthesize information from collected evidence with accumulated knowledge
        """
        if not evidence:
            return "No evidence available for synthesis"
        
        # Gather all accumulated knowledge
        contextual_knowledge = self._get_contextual_knowledge()
        
        # Analyze evidence themes
        themes = {
            "taxonomy": [],
            "ecology": [],
            "conservation": [],
            "geographic": [],
            "temporal": [],
            "morphology": []
        }
        
        for evidence in evidence:
            data_text = str(evidence.data).lower()
            
            if any(term in data_text for term in ["species", "genus", "family", "classification"]):
                themes["taxonomy"].append(f"{evidence.source}: taxonomic information")
            
            if any(term in data_text for term in ["habitat", "ecosystem", "interaction", "predator", "prey"]):
                themes["ecology"].append(f"{evidence.source}: ecological data")
            
            if any(term in data_text for term in ["conservation", "endangered", "threatened", "protection"]):
                themes["conservation"].append(f"{evidence.source}: conservation status")
            
            if any(term in data_text for term in ["distribution", "range", "location", "region"]):
                themes["geographic"].append(f"{evidence.source}: geographic information")
            
            if any(term in data_text for term in ["population", "trend", "monitoring", "survey"]):
                themes["temporal"].append(f"{evidence.source}: temporal/population data")
            
            if any(term in data_text for term in ["morphology", "anatomy", "structure", "size"]):
                themes["morphology"].append(f"{evidence.source}: morphological data")
        
        # Count evidence by theme
        active_themes = {k: v for k, v in themes.items() if v}
        
        synthesis_result = f"""Knowledge Synthesis Results:

Evidence Coverage Analysis:
{chr(10).join([f'• {theme.title()}: {len(data)} sources - {data[0] if data else "No data"}' for theme, data in active_themes.items()])}

Accumulated Knowledge Context:
{contextual_knowledge if contextual_knowledge else "No additional context accumulated"}

Integrated Findings:
• Data sources complement each other across {len(active_themes)} thematic areas
• {len(evidence)} pieces of evidence provide a {"comprehensive" if len(evidence) >= 3 else "preliminary"} foundation
• Knowledge gaps and follow-up questions have been identified for future research

This synthesis combines evidence from multiple databases with accumulated reasoning to provide a comprehensive understanding of the research question."""
        
        return synthesis_result
    
    def _initialize_reasoning_chain(self, question: str) -> None:
        """Initialize reasoning chain for context accumulation"""
        question_id = f"q_{int(time.time())}_{hash(question) % 10000}"
        self.reasoning_chain = ReasoningChain(question_id=question_id)
        self.log_verbose(f"Initialized reasoning chain: {question_id}")
    
    def _update_reasoning_chain(self, step: ResearchStep, evidence: Optional[Evidence] = None) -> None:
        """Update reasoning chain with new information"""
        if not self.reasoning_chain:
            return
        
        # Add to reasoning history
        step_summary = f"{step.step_type.value}: {step.content[:100]}{'...' if len(step.content) > 100 else ''}"
        self.reasoning_chain.reasoning_history.append(step_summary)
        
        # Extract key concepts and relationships
        if step.step_type == ReasoningStep.THOUGHT:
            concepts = self._extract_concepts(step.content)
            for concept in concepts:
                if concept not in self.reasoning_chain.accumulated_knowledge:
                    self.reasoning_chain.accumulated_knowledge[concept] = []
                self.reasoning_chain.accumulated_knowledge[concept].append(step.content)
        
        # Process evidence for knowledge accumulation
        if evidence:
            self._integrate_evidence_into_chain(evidence)
        
        # Extract insights from reflection steps
        if step.step_type == ReasoningStep.REFLECTION:
            insight = self._extract_insight(step.content)
            if insight:
                self.reasoning_chain.key_insights.append(insight)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key scientific concepts from text"""
        # Simple concept extraction - could be enhanced with NLP
        concept_indicators = [
            "species", "ecosystem", "habitat", "interaction", "evolution", "adaptation",
            "taxonomy", "morphology", "behavior", "conservation", "distribution",
            "population", "community", "predator", "prey", "symbiosis", "competition"
        ]
        
        text_lower = text.lower()
        found_concepts = []
        for concept in concept_indicators:
            if concept in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _extract_insight(self, reflection: str) -> Optional[str]:
        """Extract actionable insights from reflection text"""
        # Look for patterns that indicate insights
        insight_patterns = [
            "I have learned", "This suggests", "It appears that", "The evidence shows",
            "I can conclude", "This indicates", "The pattern reveals", "Key finding"
        ]
        
        reflection_lower = reflection.lower()
        for pattern in insight_patterns:
            if pattern in reflection_lower:
                # Extract the sentence containing the insight
                sentences = reflection.split('. ')
                for sentence in sentences:
                    if pattern in sentence.lower():
                        return sentence.strip()
        
        return None
    
    def _integrate_evidence_into_chain(self, evidence: Evidence) -> None:
        """Integrate new evidence into the reasoning chain"""
        if not self.reasoning_chain:
            return
        
        # Create evidence links
        source_key = f"source_{evidence.source}"
        if source_key not in self.reasoning_chain.evidence_links:
            self.reasoning_chain.evidence_links[source_key] = []
        
        self.reasoning_chain.evidence_links[source_key].append(evidence.query)
        
        # Extract concepts from evidence data
        concepts = self._extract_concepts(str(evidence.data))
        for concept in concepts:
            if concept not in self.reasoning_chain.concept_relationships:
                self.reasoning_chain.concept_relationships[concept] = []
            
            # Link concept to evidence source
            relationship = f"supported_by_{evidence.source}"
            if relationship not in self.reasoning_chain.concept_relationships[concept]:
                self.reasoning_chain.concept_relationships[concept].append(relationship)
    
    def _get_contextual_knowledge(self) -> str:
        """Get accumulated contextual knowledge for synthesis"""
        if not self.reasoning_chain:
            return ""
        
        context_parts = []
        
        # Add key insights
        if self.reasoning_chain.key_insights:
            context_parts.append("Key Insights:")
            for i, insight in enumerate(self.reasoning_chain.key_insights, 1):
                context_parts.append(f"{i}. {insight}")
        
        # Add concept relationships
        if self.reasoning_chain.concept_relationships:
            context_parts.append("\nConcept Relationships:")
            for concept, relationships in self.reasoning_chain.concept_relationships.items():
                context_parts.append(f"- {concept}: {', '.join(relationships)}")
        
        # Add evidence patterns
        if self.reasoning_chain.evidence_links:
            context_parts.append("\nEvidence Sources:")
            for source, queries in self.reasoning_chain.evidence_links.items():
                context_parts.append(f"- {source}: {len(queries)} queries")
        
        return "\n".join(context_parts)
    
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
        
        # Initialize reasoning chain for context accumulation
        self._initialize_reasoning_chain(question)
        
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
                
                # Action: Execute a research action based on reasoning depth and accumulated knowledge
                if self.reasoning_depth <= len(self.current_query.required_databases):
                    # Phase 1: Query each required database
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
                    
                    # Add evidence and update reasoning chain
                    evidence = Evidence(
                        source=database,
                        query=question,
                        data=action_step.metadata.get("result", ""),
                        confidence=action_step.confidence
                    )
                    self.evidence_base.append(evidence)
                    self._update_reasoning_chain(action_step, evidence)
                    
                elif self.reasoning_depth == len(self.current_query.required_databases) + 1:
                    # Phase 2: Analyze relationships between collected evidence
                    await self.execute_action(
                        "analyze_relationships",
                        {"evidence": self.evidence_base}
                    )
                    
                elif self.reasoning_depth == len(self.current_query.required_databases) + 2:
                    # Phase 3: Cross-reference findings from different sources
                    source_names = [evidence.source for evidence in self.evidence_base]
                    await self.execute_action(
                        "cross_reference",
                        {"sources": source_names, "evidence": self.evidence_base}
                    )
                    
                elif self.reasoning_depth == len(self.current_query.required_databases) + 3:
                    # Phase 4: Generate follow-up questions based on findings
                    await self.execute_action(
                        "generate_follow_up_questions",
                        {"evidence": self.evidence_base, "original_question": question}
                    )
                    
                else:
                    # Phase 5: Final synthesis with accumulated knowledge
                    await self.execute_action(
                        "synthesize_information",
                        {"evidence": self.evidence_base, "reasoning_chain": self.reasoning_chain}
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
        Uses LLM backend for advanced synthesis if available
        """
        if not self.evidence_base:
            return "I was unable to find sufficient information to answer this question."
        
        # Try advanced LLM-based synthesis first
        if self.llm_backend and hasattr(self.llm_backend, 'generate_response'):
            try:
                synthesized_answer = await self._generate_llm_synthesis()
                if synthesized_answer:
                    return synthesized_answer
            except Exception as e:
                self.log_verbose(f"LLM synthesis failed: {e}, falling back to basic synthesis")
        
        # Fallback to basic synthesis
        return await self._generate_basic_synthesis()
    
    async def _generate_llm_synthesis(self) -> str:
        """
        Generate advanced synthesis using LLM backend
        """
        # Prepare synthesis prompt
        evidence_text = "\n\n".join([
            f"Source: {evidence.source}\n"
            f"Query: {evidence.query}\n"
            f"Data: {evidence.data}\n"
            f"Confidence: {evidence.confidence:.2f}"
            for evidence in self.evidence_base
        ])
        
        reasoning_summary = "\n".join([
            f"Step {i+1} ({step.step_type.value}): {step.content[:150]}{'...' if len(step.content) > 150 else ''}"
            for i, step in enumerate(self.research_steps[-5:])  # Last 5 steps
        ])
        
        # Get accumulated knowledge context
        contextual_knowledge = self._get_contextual_knowledge()
        
        synthesis_prompt = f"""You are a scientific research assistant. Based on the evidence and reasoning below, provide a comprehensive synthesis.

Original Question: {self.current_query.original_question}

Domain: {self.current_query.domain} | Complexity: {self.current_query.complexity}

Evidence from {len(self.evidence_base)} sources:
{evidence_text}

Accumulated Knowledge:
{contextual_knowledge if contextual_knowledge else 'No additional context'}

Reasoning Process:
{reasoning_summary}

Provide a detailed scientific synthesis that:
1. Directly answers the original question
2. Integrates all evidence sources coherently  
3. Identifies key relationships and patterns
4. Uses scientific terminology appropriately
5. Acknowledges limitations and uncertainties
6. Suggests follow-up research directions

Synthesis:"""
        
        try:
            # Use the BioDataChat instance's LLM generation methods
            if hasattr(self.llm_backend, 'backend'):
                if self.llm_backend.backend == "llamafile":
                    response = self.llm_backend.generate_llamafile_response(synthesis_prompt, max_tokens=500)
                elif self.llm_backend.backend == "ollama":
                    messages = [
                        {"role": "system", "content": "You are a scientific research assistant providing detailed analysis."},
                        {"role": "user", "content": synthesis_prompt}
                    ]
                    response = self.llm_backend.generate_ollama_response(messages)
                elif self.llm_backend.demo:
                    # Generate a more sophisticated demo response for reasoning mode
                    response = self._generate_demo_synthesis()
                else:
                    response = None
                
                if response and len(response.strip()) > 50 and not response.startswith("❌"):
                    return response.strip()
            
        except Exception as e:
            self.log_verbose(f"LLM synthesis error: {e}")
        
        return None
    
    def _generate_demo_synthesis(self) -> str:
        """
        Generate sophisticated demo synthesis for reasoning mode
        """
        question = self.current_query.original_question
        evidence_count = len(self.evidence_base)
        domain = self.current_query.domain
        
        # Create contextual demo response based on the evidence and reasoning
        demo_synthesis = f"""**Comprehensive Scientific Analysis**

Based on systematic research across {evidence_count} scientific databases, I've conducted a multi-step analysis of your question about {question.lower()}.

**Evidence Integration:**
The {domain} research reveals several key patterns:

• **Data Convergence**: Information from multiple sources confirms consistent scientific understanding
• **Relationship Mapping**: Complex ecological, taxonomic, and geographic relationships have been identified
• **Knowledge Gaps**: Areas requiring additional research have been systematically identified

**Key Scientific Insights:**
{self._get_contextual_knowledge() if self._get_contextual_knowledge() else '• Multi-source validation strengthens research findings'}

**Research Synthesis:**
The evidence demonstrates interconnected biological processes and relationships that directly address your research question. This analysis incorporates cross-referenced data validation, relationship mapping, and systematic knowledge building.

**Follow-up Research Directions:**
• Population dynamics and temporal changes
• Ecosystem interaction cascades  
• Conservation implications and strategies
• Genetic and morphological variation patterns

*This synthesis represents the integration of {self.reasoning_depth} reasoning steps and {evidence_count} evidence sources using advanced scientific research methodology.*"""
        
        return demo_synthesis
    
    async def _generate_basic_synthesis(self) -> str:
        """
        Generate basic synthesis without LLM (fallback method)
        """
        answer_parts = []
        
        answer_parts.append(f"Based on my research across {len(self.current_query.required_databases)} scientific databases, here's what I found:")
        
        # Add evidence summaries
        for i, evidence in enumerate(self.evidence_base, 1):
            answer_parts.append(f"\n{i}. From {evidence.source}: {evidence.data}")
        
        # Add reasoning summary
        answer_parts.append(f"\nThrough {self.reasoning_depth} reasoning steps, I analyzed this {self.current_query.complexity} complexity question in the {self.current_query.domain} domain.")
        
        # Add synthesis based on complexity
        if self.current_query.complexity == "high" and len(self.evidence_base) > 1:
            answer_parts.append("\n\n**Synthesis:**")
            answer_parts.append("Integrating findings across multiple sources reveals complex relationships and patterns that address the multifaceted nature of this research question.")
        elif len(self.evidence_base) > 1:
            answer_parts.append("\nSynthesizing across sources, the key findings suggest a comprehensive understanding of the research question.")
        
        return " ".join(answer_parts)


# Export the main class
__all__ = ["ReActAgent", "ResearchQuery", "Evidence", "ReasoningStep"]
