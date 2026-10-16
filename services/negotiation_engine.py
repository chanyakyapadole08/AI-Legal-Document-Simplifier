NEGOTIATION_RULES = {
"Unlimited liability": (
"Request a reasonable liability cap and limit liability to direct losses caused by proven breach, negligence, fraud, or wilful misconduct."
),


"Termination without notice": (
    "Request at least thirty (30) days prior written notice, a reasonable opportunity to cure, and immediate termination only for proven gross misconduct."
),

"Immediate termination without notice": (
    "Request written notice, a fair disciplinary process, and immediate termination only for proven gross misconduct, fraud, or wilful breach."
),

"Unfair termination grounds": (
    "Ask for clear and objective termination grounds, written notice, and an opportunity to cure before termination."
),

"Unilateral termination": (
    "Request mutual termination rights, defined termination grounds, prior written notice, and an opportunity to cure."
),

"Sole discretion termination": (
    "Replace sole discretion termination with objective termination grounds and prior written notice."
),

"Sole discretion forfeiture": (
    "Ask that forfeiture apply only after written notice, a cure period, and proof of actual loss."
),

"Unilateral amendment": (
    "Require prior written consent of both parties for any material amendment or modification."
),

"Unilateral modification": (
    "Require mutual written consent before any material change to the contract, salary, duties, rights, or obligations."
),

"Unilateral policy modification": (
    "Request advance written notice and employee consent for any material change affecting employment rights, salary, duties, or obligations."
),

"Broad unilateral amendment right": (
    "Limit amendment rights by requiring written notice, mutual consent, and clear effective dates."
),

"Irrevocable rights transfer": (
    "Limit the transfer or license by purpose, duration, territory, revocability, and scope of work."
),

"Broad IP ownership": (
    "Limit company ownership to work created during employment using company resources or related to assigned duties, and exclude pre-existing or independently created intellectual property."
),

"Overbroad IP ownership": (
    "Request a clear carve-out for pre-existing intellectual property, personal projects, and independently created work that is unrelated to employment and uses no company resources."
),

"Irrevocable IP assignment": (
    "Request that IP assignment be limited to employment-related work and exclude personal or pre-existing intellectual property."
),

"Uncapped indemnity": (
    "Make indemnity mutual and subject to a reasonable cap, except for fraud, wilful misconduct, or liability that cannot be limited by law."
),

"No compensation": (
    "Request payment of accrued salary, benefits, reimbursements, and legally payable dues up to the effective termination date."
),

"Salary withholding for unmet targets": (
    "Request that earned salary not be withheld for missed targets. Separate performance incentives from earned wages and define any performance consequences clearly and proportionately."
),

"Worldwide non-compete": (
    "Limit the non-compete by time, geography, and business scope, or replace it with a narrower non-solicitation and confidentiality clause."
),

"Overbroad non-compete": (
    "Request a narrower non-compete limited by reasonable duration, geographic scope, and specific competing activities."
),

"Long non-compete": (
    "Replace the blanket non-compete with a limited non-solicitation clause and protect only confidential information and active client relationships."
),

"Client non-solicit": (
    "Limit the non-solicitation period, define restricted clients clearly, and allow ordinary professional communication not intended to solicit business."
),

"Employee penalty": (
    "Request that any penalty or bond be reduced to actual documented loss or proportionate training cost and not operate as an excessive penalty."
),

"Training cost recovery": (
    "Limit recovery to actual documented training expenses and apply it proportionately based on the remaining service period."
),

"Management calculated damages": (
    "Require damages to be based on actual proven loss and not solely calculated by management."
),

"Excessive interest": (
    "Negotiate interest below a reasonable commercial rate and apply it only after written notice and a grace period."
),

"Broad confidentiality duration": (
    "Define confidential information clearly, add reasonable exceptions, and limit confidentiality to three (3) years except for trade secrets."
),

"Unlimited confidentiality": (
    "Add reasonable exceptions and limit confidentiality duration, except for legally protected trade secrets."
),

"No claim allowed": (
    "Remove broad waiver language and preserve the right to raise genuine legal claims."
),

"Absolute right": (
    "Replace absolute discretion with reasonable discretion exercised in good faith and subject to written notice."
),

"Sole arbitrator with binding award": (
    "Request a neutral and mutually agreed arbitrator, with a fair appointment process and appropriate safeguards for an impartial arbitration."
),

"Company disclaims all liability for employee's unauthorized acts": (
    "Request a balanced liability provision that clearly distinguishes employee misconduct from acts outside the company's control and avoids an unlimited or one-sided disclaimer."
),

"Biometric monitoring without safeguards": (
    "Require informed consent, purpose limitation, data security safeguards, retention limits, and compliance with applicable data protection laws."
),

"Overbroad employee surveillance": (
    "Limit monitoring to company systems, lawful business purposes, written notice, and reasonable privacy safeguards."
),

"Broad sensitive data processing": (
    "Restrict sensitive data processing to lawful, necessary, and consent-based purposes with security safeguards."
),

"Third-party data sharing without notice": (
    "Require prior notice, purpose limitation, confidentiality obligations, and data protection safeguards before sharing personal data."
),

"Indefinite data retention": (
    "Limit data retention to the period required by law or legitimate business purpose."
)
}

# Topic rules: used when the exact issue label is not in NEGOTIATION_RULES
# (e.g. the LLM names the issue differently). Best keyword match wins.
TOPIC_RULES = [
    (("non-compete", "not to compete", "compete with", "competing business", "restraint of trade", "restrictive covenant"),
     "Ask to limit the non-compete by time (6-12 months), place and type of work, or replace it with a narrower non-solicitation and confidentiality clause."),
    (("non-solicit", "solicit", "poach"),
     "Limit the non-solicitation period, name the restricted clients or staff clearly, and allow normal professional contact that is not meant to win business."),
    (("intellectual property", "invention", "copyright", "work product", "patent", "source code"),
     "Limit company ownership to work made for the company during employment, and exclude your earlier work and personal projects that use no company resources."),
    (("confidential", "trade secret", "non-disclosure", "proprietary information"),
     "Define what counts as confidential, add standard exceptions (public or already known information, legal disclosure), and set a fixed period except for trade secrets."),
    (("liquidated damages", "bond", "penalty", "training cost", "recover", "reimburse the company", "service agreement period"),
     "Ask that any bond or repayment equal only the actual, documented cost, reduce it step by step as the service period passes, and drop fixed penalty amounts."),
    (("indemnif", "hold harmless", "liable for", "liability"),
     "Ask for a cap on liability, cover only proven losses from your own wilful misconduct or negligence, and make indemnity mutual."),
    (("terminat", "dismiss", "discharge", "retrench", "without notice", "notice period", "garden leave"),
     "Ask for a written notice period (such as 30 days) on both sides, clear grounds for termination, and a chance to fix the problem first. Keep immediate termination only for proven serious misconduct."),
    (("salary", "wages", "remuneration", "compensation", "bonus", "incentive", "withhold", "deduct", "increment", "payment"),
     "Ask that earned salary and dues are never withheld or reduced without written reasons. Keep performance pay separate from fixed salary and state the payment dates."),
    (("probation", "confirmation"),
     "Ask for a fixed probation period, written review criteria, and written confirmation by a set date."),
    (("leave", "working hours", "overtime", "holiday", "shift"),
     "Ask for stated working hours, paid leave and overtime rules in line with the law, and written approval limits for schedule changes."),
    (("transfer", "relocat", "place of work", "posting", "deputation"),
     "Ask for reasonable advance notice, relocation support, and your consent for transfers outside the current city or role."),
    (("amend", "modify", "modification", "change the terms", "policy", "alter"),
     "Ask that any material change needs your written consent and advance notice, with a clear effective date."),
    (("arbitrat", "jurisdiction", "governing law", "dispute", "court"),
     "Ask for a neutral, mutually agreed arbitrator, a convenient venue, and mediation first. Keep your right to approach courts for statutory claims."),
    (("monitor", "surveillance", "biometric", "personal data", "privacy", "data protection"),
     "Ask that monitoring is limited to company systems and lawful business use, with written notice, your consent where needed, and limits on how long data is kept."),
    (("moonlight", "outside work", "exclusive service", "other employment", "full time and attention"),
     "Ask to allow work that does not compete or clash with your job, and to need only written approval, not blanket permission, for outside activity."),
    (("resign", "notice pay", "buy out", "buyout"),
     "Ask for a short, mutual notice period, an option to pay in lieu of notice, and release of your documents and dues on exit."),
    (("assign", "delegate", "subcontract"),
     "Ask that rights and duties cannot be transferred without both sides' written consent."),
]


def _topic_advice(text: str):
    best, best_score = None, 0
    for keywords, advice in TOPIC_RULES:
        score = sum(1 for k in keywords if k in text)
        if score > best_score:
            best, best_score = advice, score
    return best


def suggest_negotiation(risk_result: dict, clause_text: str = "") -> str:
    """
    Return clause-specific negotiation guidance:
    1) exact issue rule, 2) topic rule from issue/category/clause text,
    3) generic by risk level.
    """
    issue = str(risk_result.get("detected_issue", "")).strip()

    if issue in NEGOTIATION_RULES:
        return NEGOTIATION_RULES[issue]

    haystack = " ".join([
        issue,
        str(risk_result.get("category", "")),
        str(clause_text or "")[:1500],
    ]).lower()

    topic = _topic_advice(haystack)
    if topic:
        return topic

    risk_level = str(risk_result.get("risk_level", "LOW")).upper()

    if risk_level == "HIGH":
        return (
            "Negotiate this clause before signing by requesting "
            "clear obligations, objective standards, fair remedies, "
            "reasonable limits, and written approval for material changes."
        )

    if risk_level == "MEDIUM":
        return (
            "Ask for clearer wording, objective standards, reasonable "
            "notice, and a documented approval or review process."
        )

    return (
        "This clause appears generally standard, but the commercial "
        "terms should still be reviewed before signing."
    )
