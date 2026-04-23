import json
import logging
from typing import Dict

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
) -> Dict[str, any]:
    """
    Evaluates the RAG pipeline output using the LLM-as-Judge approach.
    Uses the o1 model (e.g., o1-preview) which has specific API requirements.

    Criteria:
    1. 정확성 (Accuracy)
    2. 논리 일관성 (Logical Consistency)
    3. 규칙 인용 적절성 (Citation Appropriateness)
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

        content = response.choices[0].message.content.strip()
        # 순수 JSON 추출을 위한 안전 장치 (만약 마크다운이 섞여있을 경우 제거)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM Judge: {content}")
        return {
            "accuracy_score": 0,
            "consistency_score": 0,
            "citation_score": 0,
            "reasoning": f"JSON Parsing Error: {str(e)}",
        }
    except Exception as e:
        logger.error(f"LLM Judge evaluation failed: {e}")
        return {
            "accuracy_score": 0,
            "consistency_score": 0,
            "citation_score": 0,
            "reasoning": f"API Error: {str(e)}",
        }
