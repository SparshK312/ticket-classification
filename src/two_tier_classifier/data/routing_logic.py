"""
Routing logic and team assignments for business categories.

Defines SLA requirements, escalation paths, and production routing rules
based on business category analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from .category_mappings import BusinessCategory, BUSINESS_CATEGORIES

class PriorityLevel(Enum):
    """Priority levels for ticket routing."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class RoutingTeam:
    """Definition of a routing team with capabilities."""
    name: str
    description: str
    business_hours: str
    escalation_path: str
    specializations: List[str]
    capacity_limit: int  # Max concurrent tickets

@dataclass
class SLARequirement:
    """SLA requirements for different priority levels."""
    response_time_minutes: int
    resolution_time_hours: int
    escalation_time_hours: int
    business_hours_only: bool

# Team definitions based on business category analysis
ROUTING_TEAMS = {
    "Application Support Team": RoutingTeam(
        name="Application Support Team",
        description="Handles Vision, AppStream, and software application issues",
        business_hours="24/7 (L1), Business Hours (L2)",
        escalation_path="Senior Application Support → Development Team",
        specializations=["Vision", "AppStream", "Fusion", "Software Issues"],
        capacity_limit=50
    ),
    
    "Business Operations Team": RoutingTeam(
        name="Business Operations Team", 
        description="Handles Back Office, financial operations, and business processes",
        business_hours="Business Hours (8AM-6PM)",
        escalation_path="Senior Business Analyst → Finance Team",
        specializations=["Back Office", "Financial Operations", "Business Processes"],
        capacity_limit=30
    ),
    
    "Payment Systems Team": RoutingTeam(
        name="Payment Systems Team",
        description="Handles Chip & Pin, payment devices, and payment processing",
        business_hours="24/7 (Critical Priority)",
        escalation_path="Senior Payment Specialist → Verifone Support",
        specializations=["Chip & Pin", "PED Devices", "Payment Processing"],
        capacity_limit=25
    ),
    
    "Order Management Team": RoutingTeam(
        name="Order Management Team",
        description="Handles Vision orders, inventory, and SKU management",
        business_hours="Business Hours Extended (7AM-8PM)",
        escalation_path="Senior Order Specialist → Business Operations",
        specializations=["Vision Orders", "Inventory", "SKU Management"],
        capacity_limit=40
    ),
    
    "Print Support Team": RoutingTeam(
        name="Print Support Team",
        description="Handles all printing issues including EOD reports",
        business_hours="Business Hours (8AM-6PM)",
        escalation_path="Senior Print Specialist → Infrastructure Team",
        specializations=["Label Printing", "EOD Reports", "Printer Hardware"],
        capacity_limit=20
    ),
    
    "Identity & Access Team": RoutingTeam(
        name="Identity & Access Team",
        description="Handles Active Directory, user accounts, and access management",
        business_hours="24/7 (L1), Business Hours (L2)",
        escalation_path="Senior Identity Specialist → Security Team",
        specializations=["Active Directory", "Account Management", "Access Control"],
        capacity_limit=35
    ),
    
    "Communications Team": RoutingTeam(
        name="Communications Team",
        description="Handles email, Google services, and communication systems", 
        business_hours="Business Hours (8AM-6PM)",
        escalation_path="Senior Communications Specialist → IT Operations",
        specializations=["Email Systems", "Google Services", "Communication Tools"],
        capacity_limit=25
    ),
    
    "Store Systems Team": RoutingTeam(
        name="Store Systems Team",
        description="Handles till operations, scanners, and store-critical systems",
        business_hours="24/7 (Critical Priority)",
        escalation_path="Senior Store Systems Specialist → Store Operations",
        specializations=["Till Systems", "Scanners", "Store Operations"],
        capacity_limit=40
    ),
    
    "Mobile Devices Team": RoutingTeam(
        name="Mobile Devices Team",
        description="Handles Zebra devices, handhelds, and mobile applications",
        business_hours="Business Hours (8AM-6PM)",
        escalation_path="Senior Mobile Specialist → Hardware Vendor",
        specializations=["Zebra Devices", "Handheld Scanners", "COPS Application"],
        capacity_limit=15
    ),
    
    "General Service Desk": RoutingTeam(
        name="General Service Desk",
        description="Handles general inquiries and unclassified issues",
        business_hours="24/7 (L1), Business Hours (L2)",
        escalation_path="L2 Support → Specialist Teams",
        specializations=["General Support", "Initial Triage", "Hardware Issues"],
        capacity_limit=60
    )
}

# SLA requirements by priority level
SLA_REQUIREMENTS = {
    PriorityLevel.CRITICAL: SLARequirement(
        response_time_minutes=15,
        resolution_time_hours=1,
        escalation_time_hours=2,
        business_hours_only=False
    ),
    
    PriorityLevel.HIGH: SLARequirement(
        response_time_minutes=30,
        resolution_time_hours=4,
        escalation_time_hours=8,
        business_hours_only=False
    ),
    
    PriorityLevel.MEDIUM: SLARequirement(
        response_time_minutes=60,
        resolution_time_hours=8,
        escalation_time_hours=16,
        business_hours_only=True
    ),
    
    PriorityLevel.LOW: SLARequirement(
        response_time_minutes=240,
        resolution_time_hours=24,
        escalation_time_hours=48,
        business_hours_only=True
    )
}

def get_routing_for_category(category: BusinessCategory) -> Dict:
    """Get complete routing information for a business category."""
    category_def = BUSINESS_CATEGORIES[category]
    team = ROUTING_TEAMS[category_def.routing_team]
    priority = PriorityLevel(category_def.priority_level)
    sla = SLA_REQUIREMENTS[priority]
    
    return {
        "business_category": category_def.name,
        "routing_team": team.name,
        "team_description": team.description,
        "priority_level": priority.value,
        "sla_hours": category_def.sla_hours,
        "response_time_minutes": sla.response_time_minutes,
        "resolution_time_hours": sla.resolution_time_hours,
        "escalation_time_hours": sla.escalation_time_hours,
        "business_hours_only": sla.business_hours_only,
        "escalation_path": team.escalation_path,
        "team_specializations": team.specializations,
        "team_capacity_limit": team.capacity_limit
    }

def get_sla_for_priority(priority: str) -> SLARequirement:
    """Get SLA requirements for a priority level."""
    return SLA_REQUIREMENTS[PriorityLevel(priority)]

def calculate_urgency_score(description: str, category: BusinessCategory) -> float:
    """Calculate urgency score based on description and category."""
    category_def = BUSINESS_CATEGORIES[category]
    base_score = {
        "CRITICAL": 1.0,
        "HIGH": 0.8, 
        "MEDIUM": 0.6,
        "LOW": 0.4
    }[category_def.priority_level]
    
    # Boost score if priority keywords are present
    description_lower = description.lower()
    priority_boost = 0.0
    
    for keyword in category_def.priority_keywords:
        if keyword.lower() in description_lower:
            priority_boost += 0.1
    
    # Cap at 1.0
    return min(1.0, base_score + priority_boost)

def get_team_workload_distribution() -> Dict[str, float]:
    """Get expected workload distribution across teams."""
    total_tickets = sum(cat_def.ticket_count for cat_def in BUSINESS_CATEGORIES.values())
    
    workload = {}
    for category, cat_def in BUSINESS_CATEGORIES.items():
        team_name = cat_def.routing_team
        percentage = (cat_def.ticket_count / total_tickets) * 100
        
        if team_name in workload:
            workload[team_name] += percentage
        else:
            workload[team_name] = percentage
    
    return workload

def validate_routing_capacity() -> Dict[str, Dict]:
    """Validate that team capacities can handle expected workload."""
    workload = get_team_workload_distribution()
    total_tickets = sum(cat_def.ticket_count for cat_def in BUSINESS_CATEGORIES.values())
    
    validation_results = {}
    
    for team_name, percentage in workload.items():
        team = ROUTING_TEAMS[team_name]
        expected_daily_tickets = (percentage / 100) * (total_tickets / 365)  # Assume yearly data
        
        # Assume average resolution time of 4 hours
        concurrent_load = expected_daily_tickets * 4 / 24
        
        capacity_utilization = concurrent_load / team.capacity_limit
        
        validation_results[team_name] = {
            "expected_daily_tickets": round(expected_daily_tickets, 1),
            "expected_concurrent_load": round(concurrent_load, 1),
            "capacity_limit": team.capacity_limit,
            "capacity_utilization": round(capacity_utilization * 100, 1),
            "status": "OK" if capacity_utilization < 0.8 else "OVERLOADED" if capacity_utilization > 1.2 else "HIGH_LOAD"
        }
    
    return validation_results