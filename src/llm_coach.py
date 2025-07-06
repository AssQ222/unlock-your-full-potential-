# src/llm_coach.py
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()                           # pobiera OPENAI_API_KEY z .env
client = OpenAI()

SYSTEM_PROMPT = """
You are a world-class strength & conditioning coach and sports nutritionist.
Generate a concise 4-week roadmap (table) based on:
- avg_sleep_hours  (float, h)
- posture_notes    (list of issues: e.g. "shoulder_asymmetry", "poor_wh_ratio")
Return Markdown table with columns: Week | Focus | Training | Mobility | Nutrition.
Use short, actionable bullet points.
"""

def build_roadmap(avg_sleep: float, issues: List[str]) -> str:
    issues_str = ", ".join(issues) if issues else "none"
    user_prompt = (
        f"My average sleep is {avg_sleep:.1f} h/night. "
        f"Posture issues: {issues_str}. "
        "Create personalised 4-week roadmap."
    )
    chat = client.chat.completions.create(
        model="gpt-4o-mini",             # lub "gpt-4o"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )
    return chat.choices[0].message.content.strip()
