"""
Original → Business category mapping utilities.

Provides a centralized mapping from raw/original categories (50+ variants)
to the 10 production Business Categories used in Level 1 routing. Includes
both exact-name mappings and robust fallback heuristics for unseen labels.
"""

from typing import Dict, Optional, List

# Canonical business category names (must match CategoryDefinition.name)
BUSINESS_CATEGORY_NAMES = {
    "Software & Application Issues",
    "Back Office & Financial",
    "Payment Processing",
    "Vision Orders & Inventory",
    "Printing Services",
    "User Account Management",
    "Email & Communications",
    "Till Operations",
    "Mobile Devices",
    "General Support",
}

# Exact raw → business mappings (expandable). Keep keys as they appear in data.
RAW_TO_BUSINESS_CATEGORY: Dict[str, str] = {
    # Till / Payments
    "Till": "Till Operations",
    "Till Operations": "Till Operations",
    "Chip & Pin": "Payment Processing",
    "Chip &amp; Pin": "Payment Processing",
    "PED": "Payment Processing",
    "Payments": "Payment Processing",

    # Vision / Orders
    "Vision": "Vision Orders & Inventory",
    "Vision Orders": "Vision Orders & Inventory",
    "Legacy Vision": "Vision Orders & Inventory",
    "Orders": "Vision Orders & Inventory",
    "Inventory": "Vision Orders & Inventory",

    # Printing
    "Printing": "Printing Services",
    "Print": "Printing Services",
    "Labels": "Printing Services",

    # Applications / EUC
    "Appstream": "Software & Application Issues",
    "End User Computing": "Software & Application Issues",
    "Personal Computing": "Software & Application Issues",
    "Fusion": "Software & Application Issues",
    "Software": "Software & Application Issues",
    "Applications": "Software & Application Issues",

    # Identity / Email
    "Active Directory": "User Account Management",
    "AD": "User Account Management",
    "Google": "Email & Communications",
    "Email": "Email & Communications",

    # Devices / Hardware
    "Zebra TC52X": "Mobile Devices",
    "Hand Held": "Mobile Devices",
    "Mobile": "Mobile Devices",
    "Hardware": "General Support",
    "Telephony": "General Support",
    "Network": "General Support",
    "Server": "General Support",
    "Infrastructure": "General Support",

    # Business ops / HR
    "Back Office": "Back Office & Financial",
    "Finance": "Back Office & Financial",
    "HR - Protime": "General Support",
    "HR - HRe": "General Support",

    # Misc
    "Inquiry / Help": "General Support",
    "Request": "General Support",
    "Other": "General Support",
}

def _heuristic_map(raw: str) -> Optional[str]:
    """Heuristic mapping using substrings for unseen or variant labels."""
    if not raw:
        return None
    s = raw.strip().lower()

    # Till / Payments
    if any(k in s for k in ["till", "pos", "checkout", "register"]):
        return "Till Operations"
    if any(k in s for k in ["chip", "pin", "ped", "verifone", "card", "contactless"]):
        return "Payment Processing"

    # Vision / Orders
    if "vision" in s or any(k in s for k in ["order", "sku", "inventory", "stock"]):
        return "Vision Orders & Inventory"

    # Printing
    if any(k in s for k in ["print", "printer", "labels", "label", "eod"]):
        return "Printing Services"

    # Applications / EUC
    if any(k in s for k in ["appstream", "fusion", "application", "software", "euc", "program", "crash"]):
        return "Software & Application Issues"

    # Identity / Email
    if any(k in s for k in ["active directory", " ad ", "ad-", "ad:", "password", "account", "mfa", "2fa"]):
        return "User Account Management"
    if any(k in s for k in ["google", "gmail", "email", "mailbox"]):
        return "Email & Communications"

    # Devices / Hardware / Infra
    if any(k in s for k in ["zebra", "handheld", "hand held", "mobile device", "tc52", "tc52x"]):
        return "Mobile Devices"
    if any(k in s for k in ["hardware", "telephony", "network", "server", "router", "switch"]):
        return "General Support"

    # Business Ops
    if any(k in s for k in ["back office", "finance", "financial", "invoice", "reconciliation"]):
        return "Back Office & Financial"

    return "General Support"

def map_raw_category(raw: Optional[str]) -> Optional[str]:
    """Map a raw category label to a business category name.

    Returns None only for null/empty inputs; otherwise returns a valid business
    category name, defaulting to "General Support" if unmapped.
    """
    if raw is None:
        return None
    if raw in RAW_TO_BUSINESS_CATEGORY:
        return RAW_TO_BUSINESS_CATEGORY[raw]
    # Try normalized exact (case-insensitive) match
    for k, v in RAW_TO_BUSINESS_CATEGORY.items():
        if k.lower() == str(raw).lower():
            return v
    # Heuristic fallback
    return _heuristic_map(str(raw))

def map_series(raw_categories: List[Optional[str]]) -> List[Optional[str]]:
    """Map a list/series of raw categories to business category names."""
    return [map_raw_category(cat) for cat in raw_categories]

def coverage_report(unique_raw_categories: List[str]) -> Dict[str, any]:
    """Compute mapping coverage stats over a set of unique raw categories."""
    total = len(unique_raw_categories)
    if total == 0:
        return {"total": 0, "mapped": 0, "coverage": 1.0, "unmapped": []}

    unmapped: List[str] = []
    mapped_count = 0
    for raw in unique_raw_categories:
        mapped = map_raw_category(raw)
        if mapped is None:
            unmapped.append(raw)
        else:
            mapped_count += 1

    return {
        "total": total,
        "mapped": mapped_count,
        "coverage": mapped_count / total if total else 1.0,
        "unmapped": unmapped,
    }



