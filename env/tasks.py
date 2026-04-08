from typing import Dict, Any
from .models import Observation

TASKS = {
    "task_1": {
        "description": "Classify email correctly (Easy)",
        "initial_state": Observation(
            email_id="email_001",
            subject="Cannot reset my password",
            sender_type="customer",
            email_body="Hi, I've been trying to reset my password but I never receive the email. Please help.",
            urgency_score=0.4,
            sentiment="frustrated",
            thread_history=[],
            current_status="unread",
            previous_actions=[]
        ),
        "target_classification": "support"
    },
    "task_2": {
        "description": "Classification + Routing + Reply (Medium)",
        "initial_state": Observation(
            email_id="email_002",
            subject="Interested in Enterprise Plan",
            sender_type="customer",
            email_body="Hello, our team of 50 is interested in upgrading to the Enterprise plan. What are the next steps to get a demo?",
            urgency_score=0.6,
            sentiment="positive",
            thread_history=[],
            current_status="unread",
            previous_actions=[]
        ),
        "target_classification": "sales",
        "target_routing": "sales_department",
        "reply_keywords": ["demo", "enterprise", "team", "upgrade"]
    },
    "task_3": {
        "description": "Full Workflow - VIP Urgency (Hard)",
        "initial_state": Observation(
            email_id="email_003",
            subject="URGENT: Production server down!",
            sender_type="VIP",
            email_body="Our main production server has been unresponsive for 10 minutes. This is heavily affecting our operations. Escalation required immediately.",
            urgency_score=0.9999,
            sentiment="angry",
            thread_history=[],
            current_status="unread",
            previous_actions=[]
        ),
        "target_classification": "support",
        "target_routing": "engineering_escalation",
        "must_escalate": True,
        "reply_keywords": ["escalate", "investigating", "immediate", "urgent"]
    }
}
