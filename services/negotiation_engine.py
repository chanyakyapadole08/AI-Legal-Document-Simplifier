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

def suggest_negotiation(risk_result: dict) -> str:
    """
    Return issue-specific negotiation guidance.
    """
    issue = str(
        risk_result.get("detected_issue", "")
    ).strip()

    if issue in NEGOTIATION_RULES:
        return NEGOTIATION_RULES[issue]

    risk_level = str(
        risk_result.get("risk_level", "LOW")
    ).upper()

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
        "terms should still be reviewed before signing.")