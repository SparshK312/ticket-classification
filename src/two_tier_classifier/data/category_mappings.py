"""
Business category definitions and mappings for Level 1 classification.

Based on analysis of 6,964 actual tickets with data-driven categories:
- 10 primary business categories covering 95%+ of tickets
- Real ticket counts from production data
- Production-ready routing logic
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class BusinessCategory(Enum):
    """Primary business categories for Level 1 routing based on actual data."""
    SOFTWARE_APPLICATION = "Software & Application Issues"
    BACK_OFFICE_FINANCIAL = "Back Office & Financial"
    PAYMENT_PROCESSING = "Payment Processing"
    VISION_ORDERS = "Vision Orders & Inventory"
    PRINTING_SERVICES = "Printing Services"
    USER_ACCOUNT_MGMT = "User Account Management"
    EMAIL_COMMUNICATIONS = "Email & Communications"
    TILL_OPERATIONS = "Till Operations"
    MOBILE_DEVICES = "Mobile Devices"
    GENERAL_SUPPORT = "General Support"

@dataclass
class CategoryDefinition:
    """Complete definition of a business category."""
    name: str
    description: str
    ticket_count: int
    percentage: float
    keywords: List[str]
    priority_keywords: List[str]
    exclusion_keywords: List[str]
    routing_team: str
    sla_hours: int
    priority_level: str

# Business category definitions based on actual ticket analysis
BUSINESS_CATEGORIES = {
    BusinessCategory.SOFTWARE_APPLICATION: CategoryDefinition(
        name="Software & Application Issues",
        description="Vision software, AppStream, End User Computing, application errors",
        ticket_count=1582,
        percentage=22.7,
        keywords=[
            "vision", "appstream", "application", "software", "app", "loading", 
            "error", "fusion", "crash", "hang", "slow", "launch", "project",
            "end user computing", "system", "program"
        ],
        priority_keywords=[
            "not working", "crashed", "error", "failed", "critical", "urgent",
            "cannot use", "blocking", "unable to launch"
        ],
        exclusion_keywords=[
            "till specific", "print specific", "payment", "account", "password"
        ],
        routing_team="Application Support Team",
        sla_hours=4,
        priority_level="HIGH"
    ),
    
    BusinessCategory.BACK_OFFICE_FINANCIAL: CategoryDefinition(
        name="Back Office & Financial",
        description="Back Office operations, financial reporting, business operations",
        ticket_count=1015,
        percentage=14.6,
        keywords=[
            "back office", "financial", "finance", "payment", "double-charged", 
            "refund", "transaction", "banking", "reconciliation", "invoice",
            "billing", "statement", "business operations"
        ],
        priority_keywords=[
            "urgent", "critical", "discrepancy", "error", "failed", "blocked",
            "missing", "incorrect", "financial impact"
        ],
        exclusion_keywords=[
            "till", "vision orders", "print", "hardware", "network"
        ],
        routing_team="Business Operations Team",
        sla_hours=2,
        priority_level="HIGH"
    ),
    
    BusinessCategory.PAYMENT_PROCESSING: CategoryDefinition(
        name="Payment Processing",
        description="Chip & Pin, card payment failures, PED devices",
        ticket_count=556,
        percentage=8.0,
        keywords=[
            "chip", "pin", "chip pin", "card payment", "ped", "payment device",
            "verifone", "terminal", "contactless", "card reader", "payment failed"
        ],
        priority_keywords=[
            "not working", "failed", "error", "critical", "urgent", "customers waiting",
            "payment declined", "device offline"
        ],
        exclusion_keywords=[
            "till banking", "software", "print", "network", "vision"
        ],
        routing_team="Payment Systems Team",
        sla_hours=1,
        priority_level="CRITICAL"
    ),
    
    BusinessCategory.VISION_ORDERS: CategoryDefinition(
        name="Vision Orders & Inventory",
        description="Vision order amendments, SKU management, inventory operations",
        ticket_count=706,  # Combined Vision (180) + Legacy Vision (526)
        percentage=10.1,
        keywords=[
            "vision", "order", "amend", "sku", "inventory", "stock", "product",
            "catalog", "supplier", "delivery", "amendment", "locked", "vision order"
        ],
        priority_keywords=[
            "urgent", "locked", "unable", "failed", "deadline", "customer waiting",
            "critical", "today", "asap"
        ],
        exclusion_keywords=[
            "till", "payment", "print", "hardware", "network", "login"
        ],
        routing_team="Order Management Team",
        sla_hours=2,
        priority_level="HIGH"
    ),
    
    BusinessCategory.PRINTING_SERVICES: CategoryDefinition(
        name="Printing Services",
        description="All printing issues including EOD reports, label printing",
        ticket_count=434,  # Combined current (334) + Legacy (100)
        percentage=6.2,
        keywords=[
            "print", "printer", "printing", "label", "eod", "report", "pdf",
            "paper", "ink", "toner", "queue", "spool", "document"
        ],
        priority_keywords=[
            "not printing", "jammed", "error", "offline", "urgent", "critical",
            "no labels", "cannot print", "eod failed"
        ],
        exclusion_keywords=[
            "till", "vision orders", "payment", "network admin", "hardware repair"
        ],
        routing_team="Print Support Team",
        sla_hours=3,
        priority_level="MEDIUM"
    ),
    
    BusinessCategory.USER_ACCOUNT_MGMT: CategoryDefinition(
        name="User Account Management",
        description="Active Directory, user accounts, permissions, 2-step verification",
        ticket_count=356,  # Combined current (332) + Legacy AD (24)
        percentage=5.1,
        keywords=[
            "active directory", "ad", "account", "user", "password", "enable", "disable",
            "2-step", "verification", "locked", "permission", "access", "new employee",
            "unlock", "reset", "group", "membership", "mfa", "2fa", "samaccountname",
            "ou", "domain", "login", "cannot login", "expired password"
        ],
        priority_keywords=[
            "locked", "account locked", "unlock account", "unlock ad account",
            "password reset", "reset password", "cannot login", "access denied",
            "urgent", "new starter", "critical", "unable to access", "mfa issue",
            "2fa issue", "add to group", "remove from group"
        ],
        exclusion_keywords=[
            "till", "vision", "print", "hardware", "network setup"
        ],
        routing_team="Identity & Access Team",
        sla_hours=2,
        priority_level="HIGH"
    ),
    
    BusinessCategory.EMAIL_COMMUNICATIONS: CategoryDefinition(
        name="Email & Communications",
        description="Google, Gmail, email issues, communication systems",
        ticket_count=363,  # Combined current (313) + Legacy Google (50)
        percentage=5.2,
        keywords=[
            "google", "gmail", "email", "communication", "duplicate", "shared mailbox",
            "group membership", "charity email", "mail"
        ],
        priority_keywords=[
            "urgent", "cannot access", "critical", "blocked", "not working"
        ],
        exclusion_keywords=[
            "till", "vision", "print", "hardware", "payment"
        ],
        routing_team="Communications Team",
        sla_hours=4,
        priority_level="MEDIUM"
    ),
    
    BusinessCategory.TILL_OPERATIONS: CategoryDefinition(
        name="Till Operations",
        description="Till crashes, till functions, scanner issues, till banking",
        ticket_count=703,  # Combined Till Operations (235) + Till (218) + Legacy Till (250)
        percentage=10.1,
        keywords=[
            "till", "scanner", "crashed", "banking", "till function", "barcode",
            "checkout", "register", "pos", "cashier", "store operations", "till locked",
            "cashier locked", "locked out till", "till account", "cashier account"
        ],
        priority_keywords=[
            "crashed", "down", "offline", "frozen", "error", "unable", "failed",
            "customers waiting", "urgent", "critical", "cashier locked", "locked out",
            "till locked", "cashier cannot login", "locked till", "till access"
        ],
        exclusion_keywords=[
            "payment device", "chip pin", "vision", "print", "network admin"
        ],
        routing_team="Store Systems Team",
        sla_hours=1,
        priority_level="CRITICAL"
    ),
    
    BusinessCategory.MOBILE_DEVICES: CategoryDefinition(
        name="Mobile Devices",
        description="Zebra TC52X, handheld devices, mobile scanners, COPS application",
        ticket_count=178,  # Combined current (122) + Legacy Zebra (56)
        percentage=2.6,
        keywords=[
            "zebra", "tc52x", "mobile", "handheld", "scanner", "cops", "gun",
            "battery", "device", "picker", "cops application"
        ],
        priority_keywords=[
            "not working", "battery", "urgent", "critical", "cannot pick",
            "device failure"
        ],
        exclusion_keywords=[
            "till", "vision", "print", "network", "login general"
        ],
        routing_team="Mobile Devices Team",
        sla_hours=4,
        priority_level="MEDIUM"
    ),
    
    BusinessCategory.GENERAL_SUPPORT: CategoryDefinition(
        name="General Support",
        description="Hardware issues, network, telephony, and miscellaneous support",
        ticket_count=489,  # Sum of remaining categories
        percentage=7.0,
        keywords=[
            "hardware", "network", "telephony", "server", "internet", "connection",
            "equipment", "maintenance", "general", "help", "support", "other",
            # Enhanced hardware keywords for better classification
            "cpu", "processor", "motherboard", "memory", "ram", "hard drive", "disk",
            "physical server", "server hardware", "router", "switch", "network equipment",
            "broken", "damaged", "faulty", "replacement", "repair", "hardware failure",
            "not turning on", "physical damage", "device not working", "equipment failure"
        ],
        priority_keywords=[
            "urgent", "critical", "down", "offline", "not working", "failed",
            # Hardware-specific priority terms
            "hardware replacement", "server down", "equipment failure",
            "physical server", "broken cpu", "hardware fault", "replace broken"
        ],
        exclusion_keywords=[],  # No exclusions for general support
        routing_team="General Service Desk",
        sla_hours=6,
        priority_level="MEDIUM"
    )
}

def get_category_by_name(name: str) -> CategoryDefinition:
    """Get category definition by name."""
    for category, definition in BUSINESS_CATEGORIES.items():
        if definition.name == name:
            return definition
    raise ValueError(f"Category not found: {name}")

def get_all_keywords() -> Dict[str, List[str]]:
    """Get all keywords organized by category."""
    return {
        cat.value: definition.keywords + definition.priority_keywords
        for cat, definition in BUSINESS_CATEGORIES.items()
    }