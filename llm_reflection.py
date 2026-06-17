"""Optional LLM reflection layer for Personal Planning Agent.

The LLM never changes task selection, time allocation, or confidence. It only
turns structured planning evidence into concise reflection for the user.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from planning_models import (
    CapacityEstimate,
    LLMReflection,
    OutcomeTrend,
    TaskLevelPlan,
    UserProfile,
)


class LLMReflectionAgent:
    """Optional LLM layer for reflection, not for changing planning decisions."""

    DEFAULT_PROVIDER = "deepseek"
    DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    @classmethod
    def build_context(
        cls,
        profile: UserProfile,
        capacity_estimate: CapacityEstimate,
        outcome_trend: OutcomeTrend,
        task_level_plan: TaskLevelPlan,
    ) -> Dict[str, object]:
        return {
            "user_profile": {
                "total_sessions": profile.total_sessions,
                "average_completion_rate": profile.average_completion_rate,
                "overplanning_frequency": profile.overplanning_frequency,
                "recent_score_trend": profile.recent_score_trend,
                "common_patterns": profile.common_patterns,
            },
            "capacity_estimate": {
                "estimated_task_capacity": capacity_estimate.estimated_task_capacity,
                "reliable_task_range": capacity_estimate.reliable_task_range,
                "confidence": capacity_estimate.confidence,
                "risk_notes": capacity_estimate.risk_notes,
            },
            "outcome_trend": {
                "total_outcomes": outcome_trend.total_outcomes,
                "recent_completion_rate": outcome_trend.recent_completion_rate,
                "average_planned_minutes": outcome_trend.average_planned_minutes,
                "average_actual_minutes": outcome_trend.average_actual_minutes,
                "average_interruptions": outcome_trend.average_interruptions,
                "average_task_switches": outcome_trend.average_task_switches,
                "average_fatigue": outcome_trend.average_fatigue,
                "planning_adjustment": outcome_trend.planning_adjustment,
                "risk_flags": outcome_trend.risk_flags,
            },
            "task_level_plan": {
                "available_minutes": task_level_plan.available_minutes,
                "work_minutes": task_level_plan.work_minutes,
                "buffer_minutes": task_level_plan.buffer_minutes,
                "selected_tasks": [
                    {
                        "task_name": task.task_name,
                        "allocated_minutes": task.allocated_minutes,
                        "order": task.order,
                        "is_partial": task.is_partial,
                    }
                    for task in task_level_plan.selected_tasks
                ],
                "deferred_tasks": [
                    {
                        "task_name": task.task_name,
                        "defer_type": task.defer_type,
                        "reason": task.reason,
                    }
                    for task in task_level_plan.deferred_tasks
                ],
                "planning_risks": task_level_plan.planning_risks,
                "plan_confidence": task_level_plan.plan_confidence,
            },
        }

    @classmethod
    def generate(
        cls,
        profile: UserProfile,
        capacity_estimate: CapacityEstimate,
        outcome_trend: OutcomeTrend,
        task_level_plan: TaskLevelPlan,
        enabled: bool = True,
    ) -> LLMReflection:
        provider = os.getenv("LLM_PROVIDER", cls.DEFAULT_PROVIDER).strip().lower()
        if provider not in {"deepseek", "openai"}:
            provider = cls.DEFAULT_PROVIDER
        model = cls._resolve_model(provider)
        context = cls.build_context(
            profile, capacity_estimate, outcome_trend, task_level_plan
        )

        if not enabled:
            return cls._fallback_reflection(
                context,
                provider=provider,
                model=model,
                source="disabled",
                error="LLM reflection is disabled in the UI.",
            )

        api_key = cls._resolve_api_key(provider)
        if not api_key:
            return cls._fallback_reflection(
                context,
                provider=provider,
                model=model,
                source="rule_based_fallback",
                error=f"{cls._api_key_name(provider)} is not set.",
            )

        try:
            if provider == "deepseek":
                llm_payload = cls._call_deepseek(api_key, model, context)
            else:
                llm_payload = cls._call_openai(api_key, model, context)
            return LLMReflection(
                enabled=True,
                provider=provider,
                model=model,
                source="llm",
                reflection=str(llm_payload.get("reflection", "")).strip(),
                risk_explanation=str(llm_payload.get("risk_explanation", "")).strip(),
                next_action=str(llm_payload.get("next_action", "")).strip(),
                follow_up_question=str(llm_payload.get("follow_up_question", "")).strip(),
            )
        except (OSError, ValueError, KeyError, urllib_error.URLError) as exc:
            return cls._fallback_reflection(
                context,
                provider=provider,
                model=model,
                source="rule_based_fallback",
                error=f"LLM reflection failed: {exc}",
            )

    @classmethod
    def _resolve_model(cls, provider: str) -> str:
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_MODEL", cls.DEFAULT_DEEPSEEK_MODEL)
        return os.getenv("OPENAI_MODEL", cls.DEFAULT_OPENAI_MODEL)

    @staticmethod
    def _resolve_api_key(provider: str) -> str:
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "").strip()
        return os.getenv("OPENAI_API_KEY", "").strip()

    @staticmethod
    def _api_key_name(provider: str) -> str:
        if provider == "deepseek":
            return "DEEPSEEK_API_KEY"
        return "OPENAI_API_KEY"

    @classmethod
    def _call_deepseek(
        cls,
        api_key: str,
        model: str,
        context: Dict[str, object],
    ) -> Dict[str, str]:
        payload = {
            "model": model,
            "messages": cls._build_chat_messages(context),
            "stream": False,
            "temperature": 0.2,
        }
        req = urllib_request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        text = cls._extract_chat_completion_text(response_payload)
        parsed = json.loads(text)
        cls._validate_reflection_payload(parsed)
        return parsed

    @classmethod
    def _call_openai(
        cls,
        api_key: str,
        model: str,
        context: Dict[str, object],
    ) -> Dict[str, str]:
        payload = {
            "model": model,
            "input": cls._build_chat_messages(context),
            "max_output_tokens": 500,
        }
        req = urllib_request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        text = cls._extract_response_text(response_payload)
        parsed = json.loads(text)
        cls._validate_reflection_payload(parsed)
        return parsed

    @staticmethod
    def _build_chat_messages(context: Dict[str, object]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an LLM reflection agent inside a personal planning system. "
            "Do not change the task plan, task order, time allocation, or confidence. "
            "Explain the existing rule-based plan using the structured context. "
            "Return only valid JSON with keys: reflection, risk_explanation, "
            "next_action, follow_up_question. Keep each value concise."
        )
        user_prompt = json.dumps(context, ensure_ascii=True, indent=2)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _validate_reflection_payload(parsed: Dict[str, object]) -> None:
        required_keys = {
            "reflection",
            "risk_explanation",
            "next_action",
            "follow_up_question",
        }
        if not required_keys.issubset(parsed):
            missing = ", ".join(sorted(required_keys - set(parsed)))
            raise ValueError(f"LLM response missing keys: {missing}")

    @staticmethod
    def _extract_response_text(response_payload: Dict[str, object]) -> str:
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = response_payload.get("output", [])
        if isinstance(output_items, list):
            text_parts: List[str] = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content in content_items:
                    if isinstance(content, dict):
                        text = content.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts).strip()

        raise ValueError("No text content found in LLM response.")

    @staticmethod
    def _extract_chat_completion_text(response_payload: Dict[str, object]) -> str:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("No choices found in chat completion response.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Invalid chat completion choice format.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("No message found in chat completion response.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("No message content found in chat completion response.")
        return content.strip()

    @classmethod
    def _fallback_reflection(
        cls,
        context: Dict[str, object],
        provider: str,
        model: str,
        source: str,
        error: Optional[str] = None,
    ) -> LLMReflection:
        plan = context["task_level_plan"]
        trend = context["outcome_trend"]
        selected = plan["selected_tasks"]
        deferred = plan["deferred_tasks"]
        selected_count = len(selected) if isinstance(selected, list) else 0
        deferred_count = len(deferred) if isinstance(deferred, list) else 0
        risk_flags = trend["risk_flags"] if isinstance(trend, dict) else []

        reflection = (
            f"The plan keeps {selected_count} selected task(s) and protects "
            f"{plan['buffer_minutes']} minutes as buffer."
        )
        if deferred_count:
            reflection += f" {deferred_count} task(s) are deferred to avoid overload."

        risk_explanation = (
            "Main risk signal: " + str(risk_flags[0])
            if risk_flags
            else "No strong historical risk signal is available yet."
        )
        next_action = "Start with the first selected task and avoid adding replacement tasks if you finish early."
        follow_up_question = "After this session, which planned task was hardest to start or finish?"

        return LLMReflection(
            enabled=source != "disabled",
            provider=provider,
            model=model,
            source=source,
            reflection=reflection,
            risk_explanation=risk_explanation,
            next_action=next_action,
            follow_up_question=follow_up_question,
            error=error,
        )


def format_llm_reflection(reflection: LLMReflection) -> str:
    error_line = f"\n- Note: {reflection.error}" if reflection.error else ""
    return "\n".join(
        [
            f"- Source: {reflection.source}",
            f"- Provider: {reflection.provider}",
            f"- Model: {reflection.model}",
            f"- Reflection: {reflection.reflection}",
            f"- Risk explanation: {reflection.risk_explanation}",
            f"- Next action: {reflection.next_action}",
            f"- Follow-up question: {reflection.follow_up_question}{error_line}",
        ]
    )
