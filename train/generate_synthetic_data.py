#!/usr/bin/env python3
"""Generate deterministic synthetic spam/ham training data."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SEED = 42

FIRST_NAMES = [
    "Alex",
    "Sam",
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Jamie",
    "Cameron",
    "Avery",
]

LAST_NAMES = [
    "Miller",
    "Schmidt",
    "Weber",
    "Clark",
    "Patel",
    "Nguyen",
    "Huber",
    "Lopez",
    "Khan",
    "Fischer",
]

COMPANIES = [
    "Northwind Labs",
    "Blue Cedar Logistics",
    "Riverview Health",
    "Alpine Data Systems",
    "Bright Harbor Media",
    "Summit Retail Group",
    "Harborline Energy",
    "Metro Civic Partners",
]

PROJECTS = [
    "Q2 onboarding rollout",
    "warehouse migration",
    "billing API cleanup",
    "security awareness campaign",
    "partner renewal review",
    "regional hiring plan",
    "support queue reduction",
    "website accessibility update",
]

DEPARTMENTS = [
    "Finance",
    "HR",
    "Operations",
    "IT",
    "Sales",
    "Customer Support",
    "Compliance",
    "Procurement",
]

DOMAINS = [
    "northwind.example",
    "alpine-mail.example",
    "harborline.example",
    "metro-partners.example",
    "brightharbor.example",
]

SPAM_DOMAINS = [
    "secure-update-access.com",
    "fast-prize-center.net",
    "gift-claim-portal.org",
    "wallet-growth-now.co",
    "delivery-auth-help.info",
]

CURRENCIES = ["USD", "EUR", "GBP"]

PRODUCTS = [
    "invoice packet",
    "meeting notes",
    "policy draft",
    "budget workbook",
    "shipment manifest",
    "vendor summary",
    "security checklist",
    "training guide",
]

SPAM_PRODUCTS = [
    "miracle weight reset",
    "crypto yield package",
    "premium casino bonus",
    "loan approval file",
    "luxury reward claim",
    "exclusive pharmacy offer",
]

HAM_CLOSINGS = [
    "Thanks",
    "Best regards",
    "Kind regards",
    "Thank you",
    "Many thanks",
]

SPAM_CLOSINGS = [
    "Act now",
    "Urgent team",
    "Security Desk",
    "Claims Department",
    "Rewards Processing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="train/synthetic_data.csv")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def amount(rng: random.Random) -> str:
    currency = rng.choice(CURRENCIES)
    value = rng.randint(75, 25000)
    cents = rng.randint(0, 99)
    return f"{currency} {value:,}.{cents:02d}"


def ticket_id(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"


def legit_url(rng: random.Random, path: str) -> str:
    return f"https://{rng.choice(DOMAINS)}/{path}/{rng.randint(10, 999)}"


def spam_url(rng: random.Random, path: str) -> str:
    return f"https://{rng.choice(SPAM_DOMAINS)}/{path}/{rng.randint(100, 9999)}"


def maybe_german_sentence(rng: random.Random) -> str:
    options = [
        "Bitte gib mir kurz Bescheid, falls du etwas aendern moechtest.",
        "Die Unterlagen sind nur zur internen Abstimmung gedacht.",
        "Wenn du Rueckfragen hast, ruf mich bitte direkt an.",
        "Ich habe die wichtigsten Punkte unten zusammengefasst.",
        "Die finale Version geht morgen an das Team.",
    ]
    return rng.choice(options)


def ham_message(rng: random.Random) -> str:
    scenario = rng.randrange(8)
    sender = person(rng)
    recipient = person(rng)
    company = rng.choice(COMPANIES)
    project = rng.choice(PROJECTS)
    department = rng.choice(DEPARTMENTS)
    close = rng.choice(HAM_CLOSINGS)
    attachment = rng.choice(PRODUCTS)

    if scenario == 0:
        subject = f"Subject: Follow-up on {project}"
        body = [
            f"Hi {recipient},",
            f"I attached the latest {attachment} for the {project}.",
            f"Please review sections 2 and 3 before our call on Thursday at {rng.randint(8, 16)}:{rng.choice(['00', '15', '30', '45'])}.",
            maybe_german_sentence(rng),
            f"{close},",
            sender,
        ]
    elif scenario == 1:
        subject = f"Subject: Internal invoice check {ticket_id(rng, 'FIN')}"
        body = [
            f"Hello {recipient},",
            f"Finance needs a confirmation that the {attachment} from {company} matches the agreed amount of {amount(rng)}.",
            "This is not a payment request. We only need your approval in the ticket before 4 PM.",
            f"Reference: {legit_url(rng, 'finance/review')}",
            f"{close},",
            f"{sender} | {department}",
        ]
    elif scenario == 2:
        subject = f"Subject: IT maintenance window for {company}"
        body = [
            f"Team,",
            f"We will patch the VPN gateway tonight between 22:00 and 23:30 CET.",
            "You may see one reconnect prompt, but no password reset is required.",
            f"Status page: {legit_url(rng, 'status/maintenance')}",
            f"{close},",
            f"{sender} | IT Operations",
        ]
    elif scenario == 3:
        subject = f"Subject: Draft agenda for {project}"
        body = [
            f"Hi {recipient},",
            f"Here is the draft agenda for the workshop with {company}.",
            "I kept the discussion focused on budget, timeline, and rollout risks.",
            maybe_german_sentence(rng),
            f"{close},",
            sender,
        ]
    elif scenario == 4:
        subject = f"Subject: Delivery confirmed for PO {ticket_id(rng, 'PO')}"
        body = [
            f"Hello {recipient},",
            f"The shipment for the replacement laptops arrived at the {department} desk this morning.",
            "Nothing is blocked. Please only sign the receiving form when you have counted all cartons.",
            f"Tracking archive: {legit_url(rng, 'logistics/archive')}",
            f"{close},",
            f"{sender} | Procurement",
        ]
    elif scenario == 5:
        subject = f"Subject: Candidate interview feedback request"
        body = [
            f"Hi {recipient},",
            f"Could you add your notes on the final interview for the {department} role by end of day?",
            "The hiring panel meets tomorrow morning and wants a single scorecard.",
            maybe_german_sentence(rng),
            f"{close},",
            f"{sender} | HR",
        ]
    elif scenario == 6:
        subject = f"Subject: Customer response prepared for case {ticket_id(rng, 'CS')}"
        body = [
            f"Hello {recipient},",
            "I drafted the reply to the customer complaint and left one comment about the refund amount.",
            f"If you approve it, I will send the final response from the shared mailbox at {rng.randint(9, 17)}:00.",
            f"Case workspace: {legit_url(rng, 'support/case')}",
            f"{close},",
            f"{sender} | Customer Support",
        ]
    else:
        subject = f"Subject: Family dinner on Sunday"
        body = [
            f"Hi {recipient},",
            "We booked the table for six at 18:30 and the restaurant asked for the final headcount tomorrow.",
            "Can you confirm whether you are bringing dessert or if I should order one there?",
            "No rush, just text me later today.",
            f"{close},",
            sender,
        ]

    return "\n".join([subject, *body])


def spam_message(rng: random.Random) -> str:
    scenario = rng.randrange(10)
    sender = person(rng)
    close = rng.choice(SPAM_CLOSINGS)

    if scenario == 0:
        subject = f"Subject: Immediate account verification needed"
        body = [
            "Dear user,",
            "Your mailbox will be suspended in less than 12 hours because the security audit found unusual activity.",
            f"Confirm your password immediately here: {spam_url(rng, 'verify/login')}",
            "Failure to act will cause permanent message loss and payroll interruption.",
            close,
        ]
    elif scenario == 1:
        subject = f"Subject: Outstanding invoice {ticket_id(rng, 'INV')}"
        body = [
            "Attention payment officer,",
            f"We have not received the overdue balance of {amount(rng)} and legal escalation starts today.",
            f"Open the secured invoice copy and pay now: {spam_url(rng, 'billing/open')}",
            "Do not reply to this notice unless payment has been completed.",
            close,
        ]
    elif scenario == 2:
        subject = f"Subject: You won the international rewards draw"
        body = [
            "Congratulations winner,",
            f"Your address was selected for a guaranteed cash transfer of {amount(rng)}.",
            "To release the prize, send your full name, mobile number, and a copy of your ID within one hour.",
            f"Claim page: {spam_url(rng, 'reward/claim')}",
            close,
        ]
    elif scenario == 3:
        subject = f"Subject: Confidential donation from a public figure"
        body = [
            f"My name is {sender} and I have chosen you to receive a private humanitarian donation of {amount(rng)}.",
            "I need a trustworthy person to move the funds quickly before the board closes the file.",
            "Respond with your direct number, occupation, and mailing address to begin the release.",
            "This transaction must remain secret.",
            close,
        ]
    elif scenario == 4:
        subject = f"Subject: Delivery failed because of missing customs fee"
        body = [
            "Parcel notification,",
            "We attempted to deliver your package but customs has paused it for a small clearance fee.",
            f"Pay the release amount now: {spam_url(rng, 'parcel/pay')}",
            "If you ignore this message the parcel will be returned tonight.",
            close,
        ]
    elif scenario == 5:
        subject = f"Subject: Grow your wallet by 300 percent this week"
        body = [
            "Smart investors are joining our private crypto circle before the next surge.",
            f"Deposit only {amount(rng)} and collect daily returns with zero risk.",
            f"Register before the timer ends: {spam_url(rng, 'wallet/join')}",
            "This invitation is limited to the first 25 responders.",
            close,
        ]
    elif scenario == 6:
        subject = f"Subject: Payroll portal revalidation"
        body = [
            "Employee notice,",
            "The payroll server changed and your salary may fail unless you re-enter your credentials now.",
            f"Use the secure employee form here: {spam_url(rng, 'payroll/reset')}",
            "Managers have already completed the process. Do not delay.",
            close,
        ]
    elif scenario == 7:
        subject = f"Subject: Exclusive pharmacy discount for {rng.choice(SPAM_PRODUCTS)}"
        body = [
            "No prescription and no waiting room.",
            "Doctors do not want you to know how simple this treatment is.",
            f"Order discreetly today and unlock the hidden bundle: {spam_url(rng, 'pharmacy/buy')}",
            "Your body will thank you in 48 hours.",
            close,
        ]
    elif scenario == 8:
        subject = f"Subject: Remote data entry job approved"
        body = [
            "We reviewed your profile and approved you for a high-paying remote position with instant start.",
            f"Earn up to {amount(rng)} every week by forwarding simple forms from home.",
            f"Submit your banking details to reserve the job slot: {spam_url(rng, 'jobs/accept')}",
            "Only a few contracts remain.",
            close,
        ]
    else:
        subject = f"Subject: Buy gift cards for the director right away"
        body = [
            "I am in a confidential meeting and cannot take calls.",
            "Purchase six gift cards in the next 30 minutes and send the codes back by reply.",
            "Keep this between us until I finish the negotiation.",
            "This is urgent and will affect the deal.",
            close,
        ]

    return "\n".join([subject, *body])


def generate_dataset(rows: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    target_ham = rows // 2
    target_spam = rows - target_ham
    dataset: list[tuple[str, int]] = []
    seen: set[str] = set()
    ham_count = 0
    spam_count = 0

    while ham_count < target_ham:
        text = ham_message(rng)
        if text not in seen:
            dataset.append((text, 0))
            seen.add(text)
            ham_count += 1

    while spam_count < target_spam:
        text = spam_message(rng)
        if text not in seen:
            dataset.append((text, 1))
            seen.add(text)
            spam_count += 1

    rng.shuffle(dataset)
    return dataset


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_dataset(args.rows, args.seed)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    ham_count = sum(1 for _, label in rows if label == 0)
    spam_count = len(rows) - ham_count
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Ham: {ham_count}")
    print(f"Spam: {spam_count}")


if __name__ == "__main__":
    main()
