import json
import logging
import re
from typing import List, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.models.rule_schema import WhistleResponse
from src.services.rag.embedding import client as openai_client
from src.services.rag.rule_retrieval import rule_retriever

logger = logging.getLogger(__name__)

MAX_SITUATION_LENGTH = 1000
_BLOCKED_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions"
    r"|forget\s+(all\s+)?above"
    r"|you\s+are\s+now"
    r"|disregard\s+(all\s+)?prior",
    re.IGNORECASE,
)


def _sanitize_situation(text: str) -> str:
    """Sanitize user situation input: enforce length and strip injection patterns."""
    text = text[:MAX_SITUATION_LENGTH]
    text = _BLOCKED_PATTERNS.sub("", text)
    return text.strip()


class JudgeAgentState(TypedDict):
    """
    Represents the state of the JudgeAgent workflow. It holds all the data
    that is passed between nodes in the graph.
    """

    # The conversation history. The last message is the user's request.
    messages: List[BaseMessage]

    # Information about the user's request (situation, rule_type).
    user_info: dict

    # A list of relevant rule documents retrieved from the RAG store.
    context: List[Document]

    # The final generated judgment in JSON format.
    final_response: str


def parse_situation(state: JudgeAgentState) -> dict:
    """
    Validates that the user_info contains a situation description.
    Extracts and logs the key information for downstream nodes.
    """
    logger.info("NODE: Parsing Situation")
    if not state.get("user_info"):
        raise ValueError("User info is missing from the state.")

    user_info = state["user_info"]
    if not user_info.get("situation_description"):
        raise ValueError("Situation description is required for judgment.")

    # Sanitize early so all downstream nodes receive clean input
    sanitized_info = {
        **user_info,
        "situation_description": _sanitize_situation(
            user_info["situation_description"]
        ),
    }

    logger.debug(f"User Info: {sanitized_info}")
    return {"user_info": sanitized_info}


def retrieve_rules_and_glossary(state: JudgeAgentState) -> dict:
    """
    Retrieves relevant rules and glossary terms using the RuleRetriever.
    Uses hybrid search combining situation-based rule search and glossary lookup.
    """
    logger.info("NODE: Retrieving Rules and Glossary")
    user_info = state["user_info"]
    situation = user_info.get("situation_description", "")
    rule_type = user_info.get("rule_type")

    logger.debug(
        f"Search params: situation={(situation or '')[:80]}..., rule_type={rule_type}"
    )

    try:
        search_results = rule_retriever.hybrid_search(
            situation=situation,
            rule_type=rule_type,
            n_rules=5,
            n_glossary=3,
        )

        # Combine rules and glossary into context
        context_docs = search_results["rules"] + search_results["glossary"]

        logger.info(
            f"Retrieved {len(search_results['rules'])} rules, "
            f"{len(search_results['glossary'])} glossary terms"
        )
        return {"context": context_docs}

    except Exception as e:
        logger.exception("Failed to retrieve rules and glossary from RAG")
        raise ValueError("Failed to retrieve rules from database") from e


def generate_judgment(state: JudgeAgentState) -> dict:
    """
    Generates the final judgment by synthesizing the user's situation
    description and the retrieved rules/glossary using an LLM.
    """
    logger.info("NODE: Generating Judgment")
    user_info = state["user_info"]
    context_docs = state["context"]

    # Separate rules and glossary from context
    rule_docs = [doc for doc in context_docs if doc.metadata.get("doc_type") == "rule"]
    glossary_docs = [
        doc for doc in context_docs if doc.metadata.get("doc_type") == "glossary"
    ]

    # Prepare rules context string
    rules_context_str = "\n\n".join(
        [
            f"Rule Type: {doc.metadata.get('rule_type', 'N/A')}\n"
            f"Article: {doc.metadata.get('article', 'N/A')}\n"
            f"Page: {doc.metadata.get('page_number', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in rule_docs
        ]
    )
    if not rules_context_str:
        rules_context_str = "No specific rules found in the database."

    # Prepare glossary context string
    glossary_context_str = ""
    if glossary_docs:
        glossary_context_str = "\n\n".join(
            [
                f"Term: {doc.metadata.get('term', 'N/A')}\n"
                f"Category: {doc.metadata.get('category', 'N/A')}\n"
                f"Content: {doc.page_content}"
                for doc in glossary_docs
            ]
        )

    # Prepare the JSON schema for the prompt
    schema_json = json.dumps(WhistleResponse.model_json_schema(), indent=2)

    # Build glossary section
    glossary_section = ""
    if glossary_context_str:
        glossary_section = f"**Related Basketball Terms:**\n{glossary_context_str}\n\n"

    # Build rule type instruction
    rule_type = user_info.get("rule_type")
    rule_type_instruction = ""
    if rule_type:
        rule_type_instruction = (
            f"Focus primarily on {rule_type} rules for this judgment.\n"
        )

    system_prompt = f"""You are an expert basketball referee and rules analyst. \
Your task is to analyze a basketball game situation and provide a clear, \
authoritative judgment based on official basketball rules.

{rule_type_instruction}
{glossary_section}**Retrieved Rules from Database:**
{rules_context_str}

**Instructions:**
1. Analyze the described situation carefully.
2. Determine whether it constitutes a violation, foul, legal play, or other.
3. Provide clear reasoning citing specific rule articles from the retrieved data.
4. Include at least one rule_reference with the exact article, page number, and
   an excerpt from the rules.
5. If relevant basketball terms appear in the glossary data, include them in
   related_terms with their definitions.
6. Write the judgment_title as a concise Korean summary of the ruling.
7. Write the reasoning and situation_summary in Korean for the end user.
8. Your final output **must** be a JSON object that strictly follows this
   Pydantic schema:

```json
{schema_json}
```

JSON Output:
"""

    situation = user_info.get("situation_description") or ""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation},
            ],
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Received an invalid or empty response from OpenAI API.")

        content = response.choices[0].message.content

        try:
            extracted_data = json.loads(content)
            validated_response = WhistleResponse.model_validate(extracted_data)
            final_response_str = validated_response.model_dump_json(indent=2)
            logger.debug(f"Generated Response: {final_response_str}")
            return {"final_response": final_response_str}
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse or validate LLM response for judgment: {e}")
            raise ValueError(
                f"LLM returned an invalid judgment object: {content}"
            ) from e

    except openai.APIError as e:
        logger.error(f"OpenAI API error during judgment generation: {e}")
        raise ValueError("Failed to generate judgment due to an API error.") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during judgment generation: {e}")
        raise


# Define the graph workflow
workflow = StateGraph(JudgeAgentState)

# Add nodes to the graph
workflow.add_node("parse", parse_situation)
workflow.add_node("retrieve", retrieve_rules_and_glossary)
workflow.add_node("generate", generate_judgment)

# Define the edges for the graph
workflow.set_entry_point("parse")
workflow.add_edge("parse", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
judge_agent_graph = workflow.compile()
