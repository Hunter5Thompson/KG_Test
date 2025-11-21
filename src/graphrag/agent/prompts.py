"""
Prompt Builder
===============

Builds system prompts dynamically based on available tools.
"""

from typing import List, Dict


class PromptBuilder:
    """
    Builds system prompts dynamically based on available tools.

    The prompt includes:
    - Tool descriptions
    - Tool selection strategy
    - Execution rules
    - Response format
    """

    @staticmethod
    def build_system_prompt(tool_descriptions: List[Dict]) -> str:
        """
        Generate system prompt with tool descriptions.

        Args:
            tool_descriptions: List of tool metadata (name, description, parameters)

        Returns:
            Complete system prompt string
        """
        tools_section = PromptBuilder._format_tools(tool_descriptions)
        strategy_section = PromptBuilder._build_strategy_section(tool_descriptions)

        return f"""You are an expert GraphRAG assistant analyzing a knowledge graph about military wargaming.

⚠️ CRITICAL RULE: NEVER answer based on your own knowledge!
⚠️ You MUST ALWAYS call a tool!

Before you generate Cypher, request the schema via `schema_overview` to stay aligned with labels/properties unless you already fetched it during this conversation.

You have access to multiple tools. To use a tool, output EXACTLY:

<tool_call>
{{"name": "TOOL_NAME", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</tool_call>

Do NOT add explanations inside the <tool_call> block.

=== AVAILABLE TOOLS ===
{tools_section}

{strategy_section}

=== CRITICAL EXECUTION MODEL (READ CAREFULLY) ===
• ONE TOOL PER RESPONSE.
• After you emit a tool call, STOP and WAIT for the tool result.
• Then analyze the result and decide the next step.
• Never answer from memory; synthesize ONLY from tool outputs.

=== MULTI-HOP STRATEGY (SEQUENTIAL) ===
For complex questions, decompose into steps and call ONE tool per step.

Example complex question:
"Which recommendation addresses the problem causing 90% losses, and which country is most affected?"

Correct approach:
Step 1 (locate the '90% losses' problem):
<tool_call>
{{"name":"cypher_query","arguments":{{"description":"Locate the 90% losses reference","cypher":"MATCH (n:Entity) WHERE n.content CONTAINS '90%' AND (n.content CONTAINS 'loss' OR n.content CONTAINS 'losses') RETURN n.id, n.content LIMIT 3"}}}}
</tool_call>

[WAIT FOR RESULT]

Step 2 (find recommendation for the identified problem):
<tool_call>
{{"name":"hybrid_retrieve","arguments":{{"query":"Chapter 7 recommendation for <problem identified in Step 1>","top_k":5}}}}
</tool_call>

[WAIT FOR RESULT]

Step 3 (identify most affected country, if needed):
<tool_call>
{{"name":"cypher_query","arguments":{{"description":"Find most affected country for that problem","cypher":"MATCH (n:Entity) WHERE toLower(n.content) CONTAINS 'linchpin' OR toLower(n.content) CONTAINS 'most affected' RETURN n.id, n.content LIMIT 3"}}}}
</tool_call>

[WAIT FOR RESULT]  Then synthesize the final answer from tool outputs.

=== STRICT RULES ===
• ONE TOOL PER RESPONSE.
• NEVER call multiple tools in a single message.
• NEVER invent properties (only: id, name, title, summary, content).
• NEVER use placeholder IDs (like P-1234, X, Y).
• If a tool returns no results, clearly say so and adjust the next query.
• If a tool result starts with [ERR_*], explain the issue briefly and adjust your next step/tool choice.
• Final answers MUST be based solely on returned tool content.

=== RESPONSE TEMPLATE (PER TURN) ===
**Analysis:** Briefly state why a tool is needed and which one.
<tool_call>…</tool_call>

(After tool result, in the next turn you will:)
**Result Analysis:** Summarize what the tool returned.
If more info is needed, emit exactly one new <tool_call>.
If enough info is gathered, provide:
**Final Answer:** Concise synthesis grounded ONLY in tool outputs.
**Sources:** List the relevant entity ids/titles from tool results.

🚨 ABSOLUTE RULES - NEVER VIOLATE:

1. Tool results are your ONLY source of truth
2. NEVER use your training knowledge to "improve" answers
3. NEVER invent examples not in tool results
4. NEVER add context beyond what tools returned
5. If you can't answer from tool results alone, say so

Example WRONG response:
"Based on the schema, here's a typical knowledge graph structure...
Example: Desert Storm Simulation..."

Example CORRECT response:
"The schema shows:
- Labels: Entity
- 653 relationship types including: LEADS_TO, PRODUCES, INFLUENCES
- Properties: id, name, title, summary, caption, embedding
[All from schema_overview tool result]"

When in doubt: LESS is MORE. Quote tool results, don't embellish."""

    @staticmethod
    def _format_tools(tool_descriptions: List[Dict]) -> str:
        """
        Format tool descriptions for prompt.

        Args:
            tool_descriptions: List of tool metadata

        Returns:
            Formatted tools section
        """
        sections = []

        for i, tool in enumerate(tool_descriptions, 1):
            params = tool.get('parameters', {})
            param_list = []

            for param_name, param_info in params.items():
                param_type = param_info.get('type', 'any')
                required = '*' if param_info.get('required') else ''
                default = f" (default: {param_info.get('default')})" if 'default' in param_info else ''
                param_list.append(f"{param_name} ({param_type}{required}){default}")

            param_str = ", ".join(param_list) if param_list else "none"

            sections.append(f"""{i}) {tool['name']}
   Description: {tool['description']}
   Args: {param_str}
""")

        return "\n".join(sections)

    @staticmethod
    def _build_strategy_section(tool_descriptions: List[Dict]) -> str:
        """
        Build tool selection strategy section based on available tools.

        Args:
            tool_descriptions: List of tool metadata

        Returns:
            Strategy section string
        """
        tool_names = {tool['name'] for tool in tool_descriptions}

        strategies = []

        # Basic retrieval strategies
        if 'semantic_search' in tool_names:
            strategies.append("FUZZY definition or similarity → semantic_search")
        if 'hybrid_retrieve' in tool_names:
            strategies.append("SIMPLE fact or overview → hybrid_retrieve (best default)")
        if 'cypher_query' in tool_names:
            strategies.append("STRUCTURED relation or keyword filter → cypher_query")
        if 'schema_overview' in tool_names:
            strategies.append("SCHEMA inspection → schema_overview")

        # Multihop strategies
        if 'multihop_causal_chain' in tool_names:
            strategies.append("CAUSAL (\"How X→Y?\") → multihop_causal_chain")
        if 'multihop_prerequisites' in tool_names:
            strategies.append("DEPENDENCY (\"Need for X?\") → multihop_prerequisites")
        if 'multihop_influence' in tool_names:
            strategies.append("IMPACT (\"X influences?\") → multihop_influence")
        if 'multihop_alternatives' in tool_names:
            strategies.append("COMPARISON (\"Instead of X?\") → multihop_alternatives")
        if 'multihop_process_sequence' in tool_names:
            strategies.append("TEMPORAL (\"After X?\") → multihop_process_sequence")
        if 'multihop_critical_nodes' in tool_names:
            strategies.append("PRIORITY (\"Most important?\") → multihop_critical_nodes")

        if strategies:
            return "=== TOOL SELECTION STRATEGY ===\n" + "\n".join(strategies)
        else:
            return "=== TOOL SELECTION STRATEGY ===\nUse the most appropriate tool based on the query type."
