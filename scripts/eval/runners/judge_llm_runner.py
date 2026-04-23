import json
import logging
from typing import Any, Dict

from src.core.constants import LLM_JUDGE_EVAL_PROMPT, LLM_JUDGE_SYSTEM_PROMPT
from src.utils.llm import chat_completion_with_retry

logger = logging.getLogger(__name__)


def evaluate_with_llm_judge(
    question: str,
    generated_answer: str,
    expected_answer: str,
    context: str = "",
    prompt_template: str = LLM_JUDGE_EVAL_PROMPT,
    system_prompt: str = LLM_JUDGE_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    Evaluates the RAG pipeline output using the LLM-as-Judge approach.
    Uses gpt-4o with json_object response format.

    Criteria:
    1. 정확성 (Accuracy)
    2. 규칙 인용 적절성 / 데이터 충실도 (도메인별 의미 차이)
    """

    prompt = prompt_template.format(
        context=context,
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        response = chat_completion_with_retry(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content")
        content = content.strip()
        # 순수 JSON 추출을 위한 안전 장치 (만약 마크다운이 섞여있을 경우 제거)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        parsed = json.loads(content.strip())
        return {
            "accuracy_score": int(parsed.get("accuracy_score", 0)),
            "citation_score": int(parsed.get("citation_score", 0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM Judge: {content}")
        return {
            "accuracy_score": 0,
            "citation_score": 0,
            "reasoning": f"JSON Parsing Error: {str(e)}",
        }
    except Exception as e:
        logger.error(f"LLM Judge evaluation failed: {e}")
        return {
            "accuracy_score": 0,
            "citation_score": 0,
            "reasoning": f"API Error: {str(e)}",
        }
