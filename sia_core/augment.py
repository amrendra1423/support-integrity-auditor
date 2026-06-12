"""Label-preserving paraphrase augmentation.

Pretrained encoders normally provide paraphrase robustness for free. The lite
(offline) backend has no pretrained knowledge, so we expose the classifier to
hand-written paraphrase templates of common support intents during training.
Augmented rows are re-scored by the *same* Stage-1 signal pipeline (no manual
labels), keeping the system fully self-supervised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

PARAPHRASES = {
    # security / fraud
    "security": [
        "I spotted a sign-in from a city I have never been to.",
        "There is a withdrawal on my card that I never approved.",
        "My account appears to be accessed by somebody else.",
        "An unknown device showed up in my trusted device list overnight.",
        "I got an odd message pretending to be from your support staff.",
        "Please freeze my profile, the recent purchases are not mine.",
        "Activity in my sign-in history is from a country I've never visited.",
        "A payment went out of my account that nobody in my family made.",
    ],
    # service-impacting technical
    "outage": [
        "The app shuts itself down whenever I open preferences.",
        "Our integration endpoint keeps returning server faults.",
        "The reporting screen never finishes loading for anyone on the team.",
        "Nothing has been copied to the cloud since yesterday morning.",
        "Sign-in is impossible even with a brand new password.",
        "Our production environment is unable to serve customers right now.",
        "The platform has been unusable for our whole company since noon.",
        "Every request to your API comes back with a failure code.",
    ],
    # billing / account friction
    "billing": [
        "My statement shows the same amount taken out two times.",
        "The money you said you returned has not appeared after a week.",
        "This month's invoice is larger than the contract amount.",
        "Saving a new card keeps getting rejected.",
        "The plan re-billed itself although I had chosen to stop it.",
        "The reset message for my password never reaches my inbox.",
        "I cannot complete the verification step since my phone was lost.",
    ],
    # outage, calm-language additions
    "outage_calm": [
        "Since this morning none of our staff can reach the platform and we are losing orders.",
        "The whole office is unable to use the system and customers are waiting.",
        "We cannot open the app on any machine today and work has stopped.",
    ],
    # trivial requests dressed in urgent language (keyword bait)
    "info_loud": [
        "URGENT URGENT! The font colour on my profile page looks slightly off, please escalate immediately!",
        "CRITICAL EMERGENCY!!! The welcome banner has a typo in it.",
        "This is urgent: the icon next to my name is the wrong shade of blue.",
        "There is no fraud or anything suspicious here, nothing urgent - I just need a copy of last month's invoice.",
        "No unauthorized activity at all, just curious about your weekend opening hours.",
    ],
    # informational
    "info": [
        "Could you tell me which city your main office is in?",
        "What time does your help desk open on weekdays?",
        "Is there a cheaper rate for charities?",
        "I'd love a walkthrough of the product for my colleagues.",
        "When will the new capabilities you announced become available?",
        "What does the plan for teams cost per seat?",
        "Please walk me through putting the newest update on my laptop.",
        "I want to swap the e-mail on file for a new one.",
        "My avatar image refuses to update, not a big deal.",
        "Kindly remove my profile and all stored information.",
    ],
}

SUBJECTS = {
    "security": ["Account safety concern", "Unexpected activity",
                 "Please review my account"],
    "outage_calm": ["Service problem", "Cannot work", "System trouble"],
    "info_loud": ["URGENT request", "Critical issue", "Important"],
    "outage": ["Service problem", "System not working", "Blocked from working"],
    "billing": ["Billing question", "Charge concern", "Payment trouble"],
    "info": ["Quick question", "General request", "Information needed"],
}
CATEGORY = {"security": "Fraud", "outage": "Technical",
            "outage_calm": "Technical", "info_loud": "General Inquiry",
            "billing": "Billing", "info": "General Inquiry"}
# Typical operational turnaround per intent (hours, sampled with noise) --
# mirrors the empirical distribution of the real data.
RT_RANGE = {"security": (8, 28), "outage": (20, 55),
            "outage_calm": (20, 55), "info_loud": (30, 60),
            "billing": (30, 60), "info": (30, 60)}
PRIORITIES = ["Low", "Medium", "High", "Critical"]
CHANNELS = ["Chat", "Email", "Web Form"]
DOMAINS = ["example.com", "example.org", "example.net",
           "enterprise.org", "company.com", "tech.io"]


def augmented_tickets(n_per_phrase: int = 3, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    rows = []
    i = 0
    for intent, phrases in PARAPHRASES.items():
        for ph in phrases:
            for prio in PRIORITIES:
              for _ in range(n_per_phrase):
                lo, hi = RT_RANGE[intent]
                rows.append({
                    "Ticket_ID": f"AUG-{i:05d}",
                    "Customer_Name": "Augmented Sample",
                    "Customer_Email": f"aug{i}@{rng.choice(DOMAINS)}",
                    "Ticket_Subject": str(rng.choice(SUBJECTS[intent])),
                    "Ticket_Description": "Hi Support, " + ph,
                    "Issue_Category": CATEGORY[intent],
                    "Priority_Level": prio,
                    "Ticket_Channel": str(rng.choice(CHANNELS)),
                    "Submission_Date": "2025-01-01",
                    "Resolution_Time_Hours": int(rng.integers(lo, hi)),
                    "Assigned_Agent": "Augmented",
                    "Satisfaction_Score": int(rng.integers(1, 6)),
                })
                i += 1
    return pd.DataFrame(rows)
