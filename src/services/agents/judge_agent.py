import json
import logging
import re
from typing import List, Optional, TypedDict

import openai
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.models.rule_schema import WhistleResponse
from src.services.rag.rule_retrieval import rule_retriever
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


class JudgmentParseError(Exception):
    """Raised when LLM response cannot be parsed into WhistleResponse after retries."""

    def __init__(self, message: str, raw_content: str = "", partial: Optional[dict] = None):
        super().__init__(message)
        self.raw_content = raw_content
        self.partial = partial or {}

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

    # Violation-type keywords extracted from situation; used as the RAG query.
    search_query: str

    # A list of relevant rule documents retrieved from the RAG store.
    context: List[Document]

    # The final generated judgment in JSON format.
    final_response: str


def _parse_llm_response(
    content: str,
    situation: str,
    system_prompt: str,
) -> WhistleResponse:
    """
    Parse and validate LLM JSON output into WhistleResponse.

    Strategy:
    1. Try direct parse + validation.
    2. On failure, send the invalid JSON back to the LLM for correction (1 retry).
    3. On second failure, return a fallback WhistleResponse.
    """
    try:
        return WhistleResponse.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as first_err:
        logger.warning("First parse attempt failed (%s). Retrying with fix prompt.", first_err)

    # Retry: ask LLM to fix the invalid JSON
    fix_prompt = (
        "The following JSON is invalid or does not match the required schema.\n"
        f"Invalid JSON:\n{content}\n\n"
        "Please return a corrected JSON object that strictly matches the schema."
    )
    try:
        retry_response = chat_completion_with_retry(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation},
                {"role": "assistant", "content": content},
                {"role": "user", "content": fix_prompt},
            ],
            response_format={"type": "json_object"},
        )
        retry_content = retry_response.choices[0].message.content or ""
        return WhistleResponse.model_validate(json.loads(retry_content))
    except (json.JSONDecodeError, ValidationError) as retry_err:
        logger.warning("Retry parse also failed (%s). Raising JudgmentParseError.", retry_err)

    try:
        parsed = json.loads(content)
        partial = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        partial = {}

    raise JudgmentParseError(
        "LLM response could not be parsed into WhistleResponse after retries.",
        raw_content=content,
        partial=partial,
    )


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


def extract_keywords(state: JudgeAgentState) -> dict:
    """
    Extracts violation-type keywords from the situation description using LLM.
    These compact keywords are used as the RAG search query instead of the
    full natural-language description, narrowing the embedding space mismatch
    between long situation text and short official rule articles.
    """
    logger.info("NODE: Extracting Keywords")
    situation = state["user_info"].get("situation_description", "")

    system_prompt = (
        "농구 경기 상황에서 규칙 위반 유형을 나타내는 핵심 키워드를 추출해줘. "
        "출력 형식: 쉼표로 구분된 한국어 키워드 3-5개 (예: 파울, 트래블링, 바이얼레이션). "
        "키워드만 출력하고 다른 텍스트는 포함하지 마세요."
    )

    response = chat_completion_with_retry(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": situation},
        ],
    )
    keywords = (response.choices[0].message.content or "").strip()
    logger.info(f"Extracted search keywords: {keywords}")
    return {"search_query": keywords}


def retrieve_rules_and_glossary(state: JudgeAgentState) -> dict:
    """
    Retrieves relevant rules and glossary terms using the RuleRetriever.
    Uses hybrid search combining situation-based rule search and glossary lookup.
    """
    logger.info("NODE: Retrieving Rules and Glossary")
    user_info = state["user_info"]
    search_query = state.get("search_query") or user_info.get("situation_description", "")
    rule_type = user_info.get("rule_type")

    logger.debug(
        f"Search params: query={(search_query or '')[:80]}..., rule_type={rule_type}"
    )

    try:
        search_results = rule_retriever.hybrid_search(
            situation=search_query,
            rule_type=rule_type,
            n_rules=8,
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

**CRITICAL CITATION RULES:**
- You MUST cite rule articles ONLY from the "Retrieved Rules from Database" section above.
- Do NOT cite, invent, or reference any rule articles not present in the retrieved data.
- Every rule_reference MUST include: exact article number, page number, and a direct excerpt.
- If retrieved rules are insufficient to make a judgment, state that explicitly instead of guessing.

**Instructions:**
1. Analyze the described situation carefully.
2. Determine whether it constitutes a violation, foul, legal play, or other.
3. Provide clear reasoning citing specific rule articles from the retrieved data.
4. Include every applicable rule_reference from the retrieved rules, each with the exact
   article number, page number, and a verbatim excerpt from the rules.
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
        response = chat_completion_with_retry(
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
        validated_response = _parse_llm_response(
            content, situation, system_prompt
        )
        final_response_str = validated_response.model_dump_json(indent=2)
        logger.debug(f"Generated Response: {final_response_str}")
        return {"final_response": final_response_str}

    except openai.APIError as e:
        logger.exception("OpenAI API error during judgment generation: %s", e)
        raise ValueError("Failed to generate judgment due to an API error.") from e
    except ValueError:
        raise
    except Exception:
        logger.exception("An unexpected error occurred during judgment generation")
        raise


# Define the graph workflow
workflow = StateGraph(JudgeAgentState)

# Add nodes to the graph
workflow.add_node("parse", parse_situation)
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("retrieve", retrieve_rules_and_glossary)
workflow.add_node("generate", generate_judgment)

# Define the edges for the graph
workflow.set_entry_point("parse")
workflow.add_edge("parse", "extract_keywords")
workflow.add_edge("extract_keywords", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
judge_agent_graph = workflow.compile()
