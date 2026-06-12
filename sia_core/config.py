"""Global configuration for the Support Integrity Auditor (SIA)."""

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]
SEVERITY_TO_INT = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
INT_TO_SEVERITY = {i: s for i, s in enumerate(SEVERITY_LEVELS)}

# A ticket is a mismatch when |inferred_severity - assigned_priority| >= DELTA_THRESHOLD
DELTA_THRESHOLD = 2

# Fusion weights (rule, embedding, resolution_time). Justified by ablation in README.
FUSION_WEIGHTS = {"rule": 0.45, "embedding": 0.35, "resolution_time": 0.20}

# Customer domain tiers (proxy for customer tier, per spec).
BUSINESS_DOMAINS = {"enterprise.org", "company.com", "tech.io"}

RANDOM_SEED = 42

# Held-out split fractions (train / val / test), stratified on mismatch label.
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Verification thresholds from the problem statement.
THRESHOLDS = {"accuracy": 0.83, "macro_f1": 0.82, "per_class_recall": 0.78}

TEXT_COLS = ["Ticket_Subject", "Ticket_Description"]
REQUIRED_COLS = [
    "Ticket_ID", "Ticket_Subject", "Ticket_Description", "Issue_Category",
    "Priority_Level", "Ticket_Channel", "Customer_Email",
]
# Resolution_Time_Hours is optional at inference time (imputed when absent).
