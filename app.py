import json
import os
import sys
from pathlib import Path
from typing import Dict

# Fix path issue
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from email_env.env.environment import EmailEnv
from email_env.env.models import Action

# Safe OpenAI import
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No OPENAI_API_KEY found. Using fallback.")
        return None

    try:
        from openai  import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        print("⚠️ OpenAI import failed:", e)
        return None


client = get_openai_client()
env = EmailEnv()


# ✅ FIXED: compatible typing
def fallback_response(email_text: str) -> Dict[str, str]:
    lower_text = email_text.lower()

    if "refund" in lower_text or "damaged" in lower_text:
        return {
            "category": "refund",
            "action_type": "auto_reply",
            "response": "Sorry your order arrived damaged. We can help with a refund right away.",
        }

    if "manager" in lower_text or "terrible" in lower_text or "complaint" in lower_text:
        return {
            "category": "complaint",
            "action_type": "escalate",
            "response": "I am escalating your concern to a manager for immediate review.",
        }

    return {
        "category": "query",
        "action_type": "auto_reply",
        "response": "Thanks for your message. Here are the details about our pricing plans.",
    }


def generate_action(email_text: str) -> Action:
    if client is None:
        return Action(**fallback_response(email_text))

    prompt = f"""
Classify and respond to this email:

{email_text}

Return ONLY valid JSON:
{{
    "category": "refund | complaint | query",
    "action_type": "auto_reply | escalate | ignore",
    "response": "text"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown if present
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        payload = json.loads(content)

    except Exception as e:
        print("⚠️ OpenAI failed, using fallback:", e)
        payload = fallback_response(email_text)

    return Action(**payload)


def run_episode() -> float:
    obs = env.reset()
    total_reward = 0.0

    print(f"\n📧 Email: {obs.email_text}")

    for step in range(3):
        action = generate_action(obs.email_text)

        print(f"\nStep {step+1}")
        print("Action:", action)

        obs, reward, done, info = env.step(action)

        print("Reward:", reward.value, "| Info:", info)

        total_reward += reward.value

        if done:
            break

    return total_reward


if __name__ == "__main__":
    scores = []

    for i in range(5):
        print(f"\n====== Episode {i+1} ======")
        score = run_episode()
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print("\n✅ Average Score:", avg_score)