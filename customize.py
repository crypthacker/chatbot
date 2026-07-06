"""
customize.py
Interactive setup script — turns the generic template into a bot
personalized for YOUR business, without hand-editing JSON.

Run this once (or any time you want to rebrand), then retrain:

    python customize.py
    python train_model.py
    streamlit run app.py
"""

import json


def ask(question, default=None):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer if answer else default


def main():
    print("=" * 60)
    print(" Let's personalize your support bot")
    print("=" * 60)
    print("Press Enter to accept the [default] shown for any question.\n")

    business_name = ask("Business name", "Your Business")
    tagline = ask("Short tagline shown under the chat title",
                  "Ask me about orders, refunds, shipping, or hours.")
    hours = ask("Business hours (as you'd say them to a customer)",
                "Monday to Saturday, 9 AM to 6 PM")
    shipping_time = ask("Typical shipping time", "3-5 business days")
    shipping_cost = ask(
        "One sentence about shipping cost (or leave blank to skip)",
        "Shipping costs depend on your location and are calculated at checkout."
    )
    refund_days = ask("Refund window in days", "30")

    with open("intents_template.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    replacements = {
        "{business_name}": business_name,
        "{hours}": hours,
        "{shipping_time}": shipping_time,
        "{shipping_cost}": shipping_cost,
        "{refund_days}": refund_days,
    }

    raw = json.dumps(data)
    for placeholder, value in replacements.items():
        raw = raw.replace(placeholder, value)
    data = json.loads(raw)

    with open("intents.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    config = {
        "business_name": business_name,
        "tagline": tagline,
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nDone! Saved personalized intents.json and config.json.")
    print("Next steps:")
    print("  python train_model.py")
    print("  streamlit run app.py")
    print(
        "\nTip: intents.json now has your details baked in as plain text, "
        "so you can keep hand-editing it directly (add more patterns, "
        "new intents, tweak wording) — just re-run train_model.py after."
    )


if __name__ == "__main__":
    main()
