#!/usr/bin/env python3
"""
ODIN Agentic Advisor v3.0 - Full Enterprise Intelligence Agent
================================================================
Betopia Group  |  Odoo 19  |  LangGraph ReAct Agent  |  63 Modules  |  140+ Models

Architecture:
  - LLM-first: GPT-4o drives ALL decisions - plans, investigates, reasons, reports
  - 27 specialized tools spanning ALL business domains
  - Generic model explorer lets the LLM query ANY Odoo model dynamically
  - Multi-company aware (20+ companies with company_id filtering)
  - Web search via Tavily for external benchmarks
  - limit=0 on aggregation tools = correct totals over ALL records

Domains Covered:
  Sales & CRM, Operations, KPI/Bonus, Accounting, Payroll, Attendance,
  HR, Procurement, Purchase Requisitions, Helpdesk, Profile Database,
  Assets, Product Lab, Budgets, Currency, Announcements

Usage:
    python odin_agentic_advisor.py
"""

import os
import json
import random
import logging
import xmlrpc.client
from typing import Dict, Any, List, TypedDict, Annotated, Literal, Optional, Sequence
from datetime import datetime, timedelta
from collections import defaultdict

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import operator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ODIN-Agent")

# ========================================================================
# 1. CONFIGURATION
# ========================================================================

ODOO_URL = os.environ.get("ODOO_URL", "").rstrip("/")
ODOO_DB = os.environ.get("ODOO_DB", "")
ODOO_USER = os.environ.get("ODOO_USERNAME", "")
ODOO_KEY = os.environ.get("ODOO_API_KEY", "")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# ========================================================================
# 2. ODOO BRIDGE (Fresh connections per call, auto-retry)
# ========================================================================

class _OdooBridge:
    """Read-only XML-RPC bridge to Odoo 19 with connection resilience."""

    def __init__(self):
        self._uid = None

    def _auth(self):
        if self._uid:
            return
        common = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/common")
        self._uid = common.authenticate(ODOO_DB, ODOO_USER, ODOO_KEY, {})
        if not self._uid:
            raise ConnectionError("Odoo auth failed")
        logger.info(f"Odoo authenticated uid={self._uid}")

    def _exec(self, model, method, args, kwargs=None):
        self._auth()
        for attempt in range(2):
            try:
                proxy = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/object")
                if kwargs:
                    return proxy.execute_kw(ODOO_DB, self._uid, ODOO_KEY, model, method, args, kwargs)
                return proxy.execute_kw(ODOO_DB, self._uid, ODOO_KEY, model, method, args)
            except Exception as e:
                err = str(e)
                if attempt == 0 and any(x in err for x in ("Idle", "Request-sent", "RemoteDisconnected", "ConnectionReset")):
                    logger.warning(f"{model}.{method} retry...")
                    continue
                raise

    def search_read(self, model, domain, fields, limit=0, order=""):
        kw = {"fields": fields}
        if limit:
            kw["limit"] = limit
        if order:
            kw["order"] = order
        try:
            return self._exec(model, "search_read", [domain], kw)
        except Exception as e:
            logger.error(f"search_read({model}): {e}")
            return []

    def search_count(self, model, domain):
        try:
            return self._exec(model, "search_count", [domain])
        except Exception as e:
            logger.error(f"search_count({model}): {e}")
            return 0

    def read_group(self, model, domain, fields, groupby):
        try:
            return self._exec(model, "read_group", [domain, fields, groupby])
        except Exception as e:
            logger.error(f"read_group({model}): {e}")
            return []

    def fields_get(self, model, attributes=None):
        attrs = attributes or ["string", "type", "required", "selection"]
        try:
            return self._exec(model, "fields_get", [], {"attributes": attrs})
        except Exception as e:
            logger.error(f"fields_get({model}): {e}")
            return {}


odoo = _OdooBridge()

# ========================================================================
# 3. HELPERS
# ========================================================================

def _s(val, fb=""):
    """Safe: Odoo returns False for empty fields."""
    return val if val and val is not False else fb

def _rn(fv):
    """Relational name from [id, name]."""
    if isinstance(fv, (list, tuple)) and len(fv) >= 2:
        return str(fv[1])
    return str(fv) if fv and fv is not False else "N/A"

def _cd(company):
    """Build company filter domain."""
    if not company or company.lower() in ("all", ""):
        return []
    try:
        cid = int(company)
        return [("company_id", "=", cid)]
    except ValueError:
        return [("company_id.name", "ilike", company)]

# ========================================================================
# 4. TOOLS - GENERIC EXPLORATION (the LLM's superpower)
# ========================================================================

@tool
def discover_installed_models(keyword: str = "") -> str:
    """Discover all installed Odoo models matching a keyword.
    Use this when you don't know which model to query.
    keyword: filter by model name substring, e.g. 'sale', 'hr', 'bp', 'kpi'.
    Leave empty to list ALL available custom models."""
    CUSTOM_MODELS = [
        # Sales & CRM
        "sale.order", "sale.order.line", "crm.lead",
        "bd.platform_source", "bd.team", "bd.profile", "bd.milestone",
        "bd.order_source", "client.reference", "incoming.query.method", "lead.status",
        # Operations
        "project.operation", "project.task", "service.type", "assign.team",
        # KPI & Bonus
        "employee.kpi", "kpi.grade", "kpi.level", "kpi.role", "kpi.line",
        "kpi.deduction", "kpi.payment", "bonus.calculation", "bonus.calculation.line",
        "employee.salaries", "sales.kpi.dashboard",
        # HR
        "hr.employee", "hr.department", "hr.contract", "hr.designation",
        "hr.resign", "hr.blood.group", "hr.religion", "hr.workforce.category",
        "hr.approved.by", "hr.announcement", "company.policy", "policy.category",
        "position.change.approver", "update.employee.position.designation",
        # Attendance
        "employee.attendance.details", "employee.attendance.operations",
        "employee.attendance.source", "employee.monthly.offday",
        "employee.monthly.offday.date", "employee.movement", "employee.movement.rule",
        "employee.raw.attendance", "employee.weekly.offday",
        "hr.employee.calendar", "hr.employee.roster", "hr.employee.roster.line",
        "hr.late.penalty", "roster.attendance", "holiday.setup", "shift.change",
        # Payroll
        "hr.payslip", "hr.payslip.run", "hr.payslip.line", "hr.payslip.input",
        "hr.salary.rule", "hr.salary.rule.category", "hr.payroll.structure",
        "hr.salary.adjustment", "hr.salary.adjustment.history",
        "hr.salary.adjustment.type", "hr.contribution.register",
        "hr.loan", "tax.slab", "tax.slab.line", "hr.tax.entry", "hr.tax.challan",
        "cost.center", "contract.gross.distribution",
        # Accounting
        "account.move", "account.move.line", "account.payment", "account.journal",
        "account.account", "account.tax", "account.fiscal.year",
        "bank.statement.upload", "bank.statement.import.row", "bank.upload.file",
        "fee.account.mapping", "fiverr.description.account.map", "profile.account.mapping",
        "account.recurring.template", "recurring.payment", "recurring.payment.line",
        # Budget
        "crossovered.budget", "crossovered.budget.lines", "account.budget.post",
        # Assets
        "account.asset.asset", "account.asset.category", "account.asset.depreciation.line",
        # Procurement
        "bp.tender", "bp.tender.category", "bp.tender.document", "bp.tender.addendum",
        "bp.tender.eval.criteria", "bp.tender.line.template", "bp.tender.required.doc",
        "bp.bid", "bp.bid.line", "bp.bid.file", "bp.bill", "bp.challan",
        "bp.award.decision", "bp.vendor.company", "bp.vendor.document", "bp.vendor.owner",
        "bp.evaluation.score", "bp.evaluation.session", "bp.ledger.entry",
        "bp.audit.log", "bp.notification", "bp.clarification.thread", "bp.clarification.message",
        # Purchase Requisition
        "cus.purchase.requisition", "requisition.order.line",
        # Helpdesk
        "ticket.helpdesk", "ticket.stage", "support.ticket",
        "helpdesk.category", "helpdesk.tag", "helpdesk.type", "team.helpdesk",
        # Profile Database (Fiverr, Upwork, etc.)
        "profile.db", "fiverr.information", "upwork.information", "kwork.information",
        "freelancer.information", "payoneer.information", "pph.information",
        "stripe.information", "wise.information", "remitly.information",
        "nsave.information", "nsave.payoneer.information", "priyo.pay.information",
        "proyo.payoneer.information", "employee.sim.info", "profile.ip.info",
        "deleted.profile.log", "deleted.record.log",
        # Marketplace Profiles
        "profile.profile", "profile.transaction",
        "marketplace.marketplace", "marketplace.particular",
        # Product Lab
        "product.lab.project", "product.lab.industry", "product.lab.technology",
        "product.lab.document.type", "product.lab.project.document",
        "product.lab.project.feature", "product.lab.project.image",
        "product.lab.project.technology",
        # Announcements
        "portal.announcement", "announcement.reaction", "announcement.read.status",
        # Dashboard
        "analytics.dashboard", "operations.analytics", "sales.analytics",
        # Chatbot
        "chatbot.config", "chatbot.message", "chatbot.tag",
        # Proposal
        "proposal.template",
        # Nexus Partnership
        "nexus.kpi", "nexus.kpi.line",
        # Standard (common)
        "res.company", "res.partner", "res.users", "res.currency",
        "product.template", "product.product",
    ]
    if keyword:
        matches = [m for m in CUSTOM_MODELS if keyword.lower() in m.lower()]
    else:
        matches = CUSTOM_MODELS
    if not matches:
        return "No models matching '{}'. Try a broader keyword.".format(keyword)
    lines = ["Available models ({} matches):".format(len(matches))]
    for m in sorted(matches):
        lines.append("  {}".format(m))
    return "\n".join(lines)


@tool
def explore_model_fields(model_name: str = "sale.order") -> str:
    """Discover all fields of ANY Odoo model. Use this FIRST when you need to query
    a model you haven't used before, to learn its field names and types.
    Returns field name, type, and label for each field."""
    fields = odoo.fields_get(model_name)
    if not fields:
        return "Model '{}' not found or has no accessible fields.".format(model_name)
    lines = ["Fields for '{}' ({} fields):".format(model_name, len(fields))]
    for fname, fdef in sorted(fields.items()):
        ftype = fdef.get("type", "?")
        label = fdef.get("string", "")
        sel = ""
        if ftype == "selection" and fdef.get("selection"):
            opts = [str(s[0]) for s in fdef["selection"][:8]]
            sel = " [{}]".format(", ".join(opts))
        lines.append("  {} ({}): {}{}".format(fname, ftype, label, sel))
    return "\n".join(lines[:150])


@tool
def query_any_model(
    model_name: str = "sale.order",
    domain_json: str = "[]",
    fields_csv: str = "name",
    limit: int = 100,
    order: str = "",
    company: str = "",
) -> str:
    """Query ANY Odoo model dynamically. The ultimate exploration tool.
    model_name: e.g. 'hr.payslip', 'bp.tender', 'ticket.helpdesk'
    domain_json: Odoo domain as JSON, e.g. '[["state","=","done"]]'
    fields_csv: comma-separated field names, e.g. 'name,amount_total,state'
    limit: max records (default 100)
    order: sort order, e.g. 'create_date desc'
    company: company name/id filter or 'all'"""
    try:
        domain = json.loads(domain_json)
    except json.JSONDecodeError:
        return "Invalid domain_json: {}".format(domain_json)
    domain += _cd(company)
    fields = [f.strip() for f in fields_csv.split(",")]
    records = odoo.search_read(model_name, domain, fields, limit=limit, order=order)
    if not records:
        return "No records found in '{}' with domain {}".format(model_name, domain_json)
    lines = ["Results from '{}' ({} records, limit={}):".format(model_name, len(records), limit)]
    for r in records[:50]:
        parts = []
        for f in fields:
            val = r.get(f, "")
            val = _rn(val) if isinstance(val, (list, tuple)) else _s(val, "-")
            parts.append("{}={}".format(f, val))
        lines.append("  [{}] {}".format(r.get("id", ""), " | ".join(parts)))
    if len(records) > 50:
        lines.append("  ... and {} more records".format(len(records) - 50))
    return "\n".join(lines)


@tool
def count_records(
    model_name: str = "sale.order",
    domain_json: str = "[]",
    company: str = "",
) -> str:
    """Count records in any Odoo model. Fast way to gauge data volume."""
    try:
        domain = json.loads(domain_json)
    except json.JSONDecodeError:
        return "Invalid domain_json."
    domain += _cd(company)
    cnt = odoo.search_count(model_name, domain)
    return "{}: {} records (domain: {}, company: {})".format(model_name, cnt, domain_json, company or "all")


@tool
def get_full_data_overview(company: str = "") -> str:
    """Get record counts across ALL major business models (30+ models).
    This is the best starting point for any analysis.
    company: filter by company name/id, or leave empty for all."""
    cd = _cd(company)
    models = {
        "sale.order": [],
        "crm.lead": [],
        "project.operation": [],
        "project.task": [],
        "employee.kpi": [],
        "bonus.calculation": [],
        "kpi.payment": [],
        "hr.employee": [("active", "=", True)],
        "hr.department": [],
        "hr.contract": [("state", "=", "open")],
        "account.move": [("state", "=", "posted")],
        "account.payment": [],
        "crossovered.budget": [],
        "account.asset.asset": [],
        "hr.payslip": [],
        "hr.payslip.run": [],
        "hr.loan": [],
        "employee.attendance.details": [],
        "hr.late.penalty": [],
        "employee.movement": [],
        "bp.tender": [],
        "bp.bid": [],
        "bp.vendor.company": [],
        "cus.purchase.requisition": [],
        "ticket.helpdesk": [],
        "profile.db": [],
        "product.lab.project": [],
        "portal.announcement": [],
        "res.currency": [("active", "=", True)],
        "res.company": [],
    }
    lines = ["Full Data Overview (company: {}):".format(company or "all")]
    for model, base_domain in models.items():
        domain = base_domain + cd
        try:
            cnt = odoo.search_count(model, domain)
            lines.append("  {}: {}".format(model, cnt))
        except Exception:
            lines.append("  {}: (not available)".format(model))
    return "\n".join(lines)


@tool
def list_companies() -> str:
    """List all companies in the Odoo instance. Use this to understand
    the multi-company structure before filtering queries."""
    companies = odoo.search_read("res.company", [], [
        "name", "partner_id", "currency_id", "country_id",
    ], order="name")
    if not companies:
        return "No companies found."
    lines = ["Companies ({}):".format(len(companies))]
    for c in companies:
        lines.append("  ID {}: {} - Currency: {}, Country: {}".format(
            c["id"], c["name"], _rn(c.get("currency_id")), _rn(c.get("country_id"))))
    return "\n".join(lines)


# ========================================================================
# 5. TOOLS - SALES & CRM (limit=0 = fetch ALL for accurate totals)
# ========================================================================

@tool
def query_sales_orders(
    status: str = "all", order_status: str = "all",
    date_from: str = "", date_to: str = "",
    limit: int = 0, company: str = "",
) -> str:
    """Query sale.order with revenue, customers, status breakdown.
    status: draft|sale|done|cancel|all (Odoo standard state).
    order_status: nra|wip|delivered|complete|cancelled|revisions|issues|all (custom delivery status).
    Dates: YYYY-MM-DD. limit: 0 means ALL records (accurate totals)."""
    domain = _cd(company)
    if status != "all":
        domain.append(("state", "=", status))
    if order_status != "all":
        domain.append(("order_status", "=", order_status))
    if date_from:
        domain.append(("date_order", ">=", date_from))
    if date_to:
        domain.append(("date_order", "<=", date_to))

    orders = odoo.search_read("sale.order", domain, [
        "name", "partner_id", "state", "order_status", "amount_total",
        "currency_id", "date_order", "invoice_status", "team_id",
        "payment_state", "company_id",
    ], limit=limit, order="date_order desc")
    if not orders:
        return "No sales orders found."

    total = sum(o.get("amount_total", 0) or 0 for o in orders)
    by_state = defaultdict(lambda: {"count": 0, "amount": 0})
    by_order_status = defaultdict(lambda: {"count": 0, "amount": 0})
    by_cust = defaultdict(float)
    by_currency = defaultdict(float)
    by_company = defaultdict(lambda: {"count": 0, "amount": 0})
    for o in orders:
        amt = o.get("amount_total", 0) or 0
        st = _s(o.get("state"), "unknown")
        by_state[st]["count"] += 1
        by_state[st]["amount"] += amt
        ost = _s(o.get("order_status"), "(blank)")
        by_order_status[ost]["count"] += 1
        by_order_status[ost]["amount"] += amt
        by_cust[_rn(o.get("partner_id"))] += amt
        by_currency[_rn(o.get("currency_id"))] += amt
        by_company[_rn(o.get("company_id"))]["count"] += 1
        by_company[_rn(o.get("company_id"))]["amount"] += amt

    top = sorted(by_cust.items(), key=lambda x: -x[1])[:10]
    lines = ["Sales Orders ({} records, company={}):".format(len(orders), company or "all"),
             "  Total Revenue: ${:,.2f}".format(total),
             "  By Order Status:"]
    for st, d in sorted(by_order_status.items(), key=lambda x: -x[1]["count"]):
        lines.append("    {}: {} orders  ${:,.2f}".format(st, d["count"], d["amount"]))
    lines.append("  By State:")
    for st, d in sorted(by_state.items()):
        lines.append("    {}: {} orders  ${:,.2f}".format(st, d["count"], d["amount"]))
    if len(by_company) > 1:
        lines.append("  By Company (top 10):")
        for co, d in sorted(by_company.items(), key=lambda x: -x[1]["amount"])[:10]:
            lines.append("    {}: {} orders  ${:,.2f}".format(co, d["count"], d["amount"]))
    lines.append("  By Currency:")
    for cur, amt in sorted(by_currency.items(), key=lambda x: -x[1]):
        lines.append("    {}: ${:,.2f}".format(cur, amt))
    lines.append("  Top 10 Customers:")
    for name, amt in top:
        lines.append("    {}: ${:,.2f}".format(name, amt))
    return "\n".join(lines)


@tool
def query_crm_leads(stage: str = "all", limit: int = 0, company: str = "") -> str:
    """Query CRM leads/opportunities. stage: all|won|lost|(name substring).
    limit=0 means ALL records."""
    domain = _cd(company)
    if stage == "won":
        domain.append(("stage_id.is_won", "=", True))
    elif stage == "lost":
        domain.append(("active", "=", False))
    elif stage != "all":
        domain.append(("stage_id.name", "ilike", stage))

    leads = odoo.search_read("crm.lead", domain, [
        "name", "partner_id", "stage_id", "expected_revenue",
        "probability", "user_id", "type",
    ], limit=limit)
    if not leads:
        return "No CRM leads found."

    total_exp = sum(l.get("expected_revenue", 0) or 0 for l in leads)
    by_stage = defaultdict(int)
    for l in leads:
        by_stage[_rn(l.get("stage_id"))] += 1

    lines = ["CRM Leads ({} records):".format(len(leads)),
             "  Expected Revenue: ${:,.2f}".format(total_exp),
             "  By Stage:"]
    for st, cnt in sorted(by_stage.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, cnt))
    return "\n".join(lines)


# ========================================================================
# 6. TOOLS - OPERATIONS (limit=0 for full data)
# ========================================================================

@tool
def query_operations(status: str = "all", limit: int = 0, company: str = "") -> str:
    """Query project.operation (delivery tracking). limit=0 = ALL records.
    status: all|nra|wip|delivered|complete|cancelled|revisions|issues."""
    domain = _cd(company)
    if status != "all":
        domain.append(("order_status", "=", status))

    ops = odoo.search_read("project.operation", domain, [
        "name", "order_status", "so_id", "employee_id", "date",
        "delivery_last_date", "delivery_amount", "monetary_value",
        "customer_id", "service_type_id", "assigned_team_id",
    ], limit=limit)
    if not ops:
        return "No project.operation records found."

    by_st = defaultdict(int)
    total_val = 0
    for o in ops:
        st = _s(o.get("order_status"), "unknown")
        by_st[st] += 1
        total_val += o.get("delivery_amount", 0) or o.get("monetary_value", 0) or 0

    delivered = by_st.get("delivered", 0) + by_st.get("complete", 0)
    rate = (delivered / len(ops) * 100) if ops else 0

    lines = ["Operations ({} records, company={}):".format(len(ops), company or "all"),
             "  Delivery Rate: {:.1f}%".format(rate),
             "  Total Value: ${:,.2f}".format(total_val),
             "  By Status:"]
    for st, cnt in sorted(by_st.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, cnt))
    return "\n".join(lines)


# ========================================================================
# 7. TOOLS - KPI & BONUS (limit=0)
# ========================================================================

@tool
def query_kpi_records(year: str = "2025", quarter: str = "", limit: int = 0, company: str = "") -> str:
    """Query employee.kpi (performance + bonus). limit=0 = ALL records.
    year: e.g. '2025', quarter: '1'-'4' or ''."""
    domain = [("year", "=", year)] + _cd(company)
    if quarter:
        domain.append(("quarter", "=", quarter))

    kpis = odoo.search_read("employee.kpi", domain, [
        "employee_id", "role_type", "grade_id", "minimum_target",
        "total_paid", "total_sales", "bonus_amount", "is_penalty",
        "state", "period_start", "period_end",
    ], limit=limit)
    if not kpis:
        return "No KPI records for year={}, quarter={}.".format(year, quarter or "all")

    total_bonus = sum(k.get("bonus_amount", 0) or 0 for k in kpis)
    total_sales = sum(k.get("total_sales", 0) or 0 for k in kpis)
    total_paid = sum(k.get("total_paid", 0) or 0 for k in kpis)
    meeting = sum(1 for k in kpis if (k.get("total_paid", 0) or 0) >= (k.get("minimum_target", 0) or 0) and (k.get("minimum_target", 0) or 0) > 0)
    penalties = sum(1 for k in kpis if k.get("is_penalty"))

    grade = defaultdict(int)
    role = defaultdict(int)
    for k in kpis:
        grade[_rn(k.get("grade_id"))] += 1
        role[_s(k.get("role_type"), "unknown")] += 1

    header = "KPI Summary - {}".format(year)
    if quarter:
        header += " Q{}".format(quarter)
    header += " ({} records):".format(len(kpis))
    lines = [header,
        "  Total Sales: ${:,.2f}".format(total_sales),
        "  Total Collected: ${:,.2f}".format(total_paid),
        "  Collection Rate: {:.1f}%".format((total_paid / max(total_sales, 1)) * 100),
        "  Meeting Target: {}/{} ({:.1f}%)".format(meeting, len(kpis), meeting / max(len(kpis), 1) * 100),
        "  Penalties: {}".format(penalties),
        "  Total Bonus: {:,.0f} BDT".format(total_bonus),
        "  Grades:"]
    for g, c in sorted(grade.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(g, c))
    lines.append("  Roles:")
    for r, c in sorted(role.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(r, c))
    return "\n".join(lines)


# ========================================================================
# 8. TOOLS - ACCOUNTING & FINANCE (limit=0 for accurate totals)
# ========================================================================

@tool
def query_accounting(move_type: str = "all", date_from: str = "", date_to: str = "",
                     limit: int = 0, company: str = "") -> str:
    """Query account.move (invoices/bills/entries). limit=0 = ALL records.
    move_type: all|out_invoice|in_invoice|entry."""
    domain = [("state", "=", "posted")] + _cd(company)
    if move_type != "all":
        domain.append(("move_type", "=", move_type))
    if date_from:
        domain.append(("date", ">=", date_from))
    if date_to:
        domain.append(("date", "<=", date_to))

    moves = odoo.search_read("account.move", domain, [
        "name", "move_type", "amount_total", "amount_residual",
        "date", "partner_id", "payment_state", "company_id",
    ], limit=limit, order="date desc")
    if not moves:
        return "No accounting entries found."

    by_type = defaultdict(lambda: {"count": 0, "total": 0, "residual": 0})
    by_company = defaultdict(float)
    for m in moves:
        mt = _s(m.get("move_type"), "other")
        by_type[mt]["count"] += 1
        by_type[mt]["total"] += m.get("amount_total", 0) or 0
        by_type[mt]["residual"] += m.get("amount_residual", 0) or 0
        by_company[_rn(m.get("company_id"))] += m.get("amount_total", 0) or 0

    inv = by_type.get("out_invoice", {}).get("total", 0)
    bills = by_type.get("in_invoice", {}).get("total", 0)
    ar = by_type.get("out_invoice", {}).get("residual", 0)
    ap = by_type.get("in_invoice", {}).get("residual", 0)

    lines = ["Accounting ({} entries, company={}):".format(len(moves), company or "all"),
             "  Invoiced: ${:,.2f} (AR: ${:,.2f})".format(inv, ar),
             "  Bills: ${:,.2f} (AP: ${:,.2f})".format(bills, ap),
             "  Gross Margin: ${:,.2f}".format(inv - bills),
             "  By Type:"]
    for mt, d in sorted(by_type.items()):
        lines.append("    {}: {}  total=${:,.2f}  residual=${:,.2f}".format(mt, d["count"], d["total"], d["residual"]))
    if len(by_company) > 1:
        lines.append("  By Company:")
        for co, amt in sorted(by_company.items(), key=lambda x: -x[1]):
            lines.append("    {}: ${:,.2f}".format(co, amt))
    return "\n".join(lines)


@tool
def query_profit_and_loss(date_from: str = "", date_to: str = "",
                          company: str = "") -> str:
    """Profit & Loss (Income Statement) from journal entries (account.move.line).
    This is the REAL P&L matching what Odoo Accounting reports show.
    Queries posted journal items grouped by account to compute:
    Income (Sales Revenue, Other Income) minus Expenses = Net Profit.
    date_from/date_to: YYYY-MM-DD (required - user must provide). company: name, id, or empty for all."""
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("date", ">=", date_from),
        ("date", "<=", date_to),
    ] + _cd(company)

    # Odoo 17+ account_type values for P&L accounts
    income_types = ["income", "income_other"]
    expense_types = ["expense", "expense_direct_cost", "expense_depreciation"]
    pl_types = income_types + expense_types

    domain.append(("account_id.account_type", "in", pl_types))

    lines_data = odoo.search_read("account.move.line", domain, [
        "account_id", "balance", "debit", "credit",
    ], limit=0)

    if not lines_data:
        return "No P&L journal entries found for {} to {} (company={}).".format(
            date_from, date_to, company or "all")

    # Get account details to classify
    account_ids = list(set(l["account_id"][0] for l in lines_data if l.get("account_id")))
    accounts = odoo.search_read("account.account", [("id", "in", account_ids)], [
        "name", "code", "account_type",
    ])
    acc_map = {a["id"]: a for a in accounts}

    # Aggregate by account
    income_accounts = defaultdict(float)
    expense_accounts = defaultdict(float)
    total_income = 0
    total_expense = 0

    for line in lines_data:
        acc_id = line["account_id"][0] if line.get("account_id") else None
        if not acc_id or acc_id not in acc_map:
            continue
        acc = acc_map[acc_id]
        acc_type = acc.get("account_type", "")
        # In Odoo, income accounts have negative balance (credit > debit)
        # We negate to show income as positive
        balance = line.get("balance", 0) or 0
        label = "{} {}".format(acc.get("code", ""), acc.get("name", ""))

        if acc_type in income_types:
            income_accounts[label] += balance
            total_income += balance
        elif acc_type in expense_types:
            expense_accounts[label] += balance
            total_expense += balance

    # Income stored as negative balance in Odoo, negate for display
    display_income = -total_income
    display_expense = total_expense
    net_profit = display_income - display_expense

    out = ["PROFIT & LOSS: {} to {} (company: {})".format(date_from, date_to, company or "all")]
    out.append("=" * 55)
    out.append("")
    out.append("  Net Profit: {:,.2f}".format(net_profit))
    out.append("")
    out.append("  INCOME: {:,.2f}".format(display_income))
    # Sort income accounts by absolute amount descending
    for label, bal in sorted(income_accounts.items(), key=lambda x: x[1]):
        out.append("    {}: {:,.2f}".format(label, -bal))
    out.append("")
    out.append("  EXPENSES: {:,.2f}".format(display_expense))
    for label, bal in sorted(expense_accounts.items(), key=lambda x: -x[1]):
        out.append("    {}: {:,.2f}".format(label, bal))

    return "\n".join(out)


@tool
def query_balance_sheet(date_to: str = "", company: str = "") -> str:
    """Balance Sheet as of a date. Shows Assets, Liabilities, Equity from journal entries.
    Matches the Odoo Balance Sheet report. date_to: YYYY-MM-DD (required - user must provide)."""
    if not date_to:
        return "Please provide date_to (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("date", "<=", date_to),
    ] + _cd(company)

    bs_types = {
        "asset_receivable": "Assets", "asset_cash": "Assets", "asset_current": "Assets",
        "asset_non_current": "Assets", "asset_prepayments": "Assets", "asset_fixed": "Assets",
        "liability_payable": "Liabilities", "liability_credit_card": "Liabilities",
        "liability_current": "Liabilities", "liability_non_current": "Liabilities",
        "equity": "Equity", "equity_unaffected": "Equity",
    }
    domain.append(("account_id.account_type", "in", list(bs_types.keys())))

    lines_data = odoo.search_read("account.move.line", domain, [
        "account_id", "balance",
    ], limit=0)
    if not lines_data:
        return "No balance sheet data found as of {} (company={}).".format(date_to, company or "all")

    account_ids = list(set(l["account_id"][0] for l in lines_data if l.get("account_id")))
    accounts = odoo.search_read("account.account", [("id", "in", account_ids)], [
        "name", "code", "account_type",
    ])
    acc_map = {a["id"]: a for a in accounts}

    sections = {"Assets": defaultdict(float), "Liabilities": defaultdict(float), "Equity": defaultdict(float)}
    totals = {"Assets": 0, "Liabilities": 0, "Equity": 0}

    for line in lines_data:
        acc_id = line["account_id"][0] if line.get("account_id") else None
        if not acc_id or acc_id not in acc_map:
            continue
        acc = acc_map[acc_id]
        section = bs_types.get(acc.get("account_type", ""), None)
        if not section:
            continue
        balance = line.get("balance", 0) or 0
        label = "{} {}".format(acc.get("code", ""), acc.get("name", ""))
        sections[section][label] += balance
        totals[section] += balance

    out = ["BALANCE SHEET as of {} (company: {})".format(date_to, company or "all")]
    out.append("=" * 55)
    for section in ["Assets", "Liabilities", "Equity"]:
        display_total = totals[section] if section == "Assets" else -totals[section]
        out.append("")
        out.append("  {}: {:,.2f}".format(section.upper(), display_total))
        items = sections[section]
        for label, bal in sorted(items.items(), key=lambda x: -abs(x[1])):
            display_bal = bal if section == "Assets" else -bal
            if abs(display_bal) > 0.01:
                out.append("    {}: {:,.2f}".format(label, display_bal))
    out.append("")
    out.append("  CHECK: Assets({:,.2f}) = Liabilities({:,.2f}) + Equity({:,.2f})".format(
        totals["Assets"], -totals["Liabilities"], -totals["Equity"]))
    return "\n".join(out)


@tool
def query_trial_balance(date_from: str = "", date_to: str = "",
                        company: str = "") -> str:
    """Trial Balance showing all accounts with debit, credit, and balance.
    Shows every account that had entries in the period.
    date_from/date_to: YYYY-MM-DD (required - user must provide)."""
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("date", ">=", date_from),
        ("date", "<=", date_to),
    ] + _cd(company)

    lines_data = odoo.search_read("account.move.line", domain, [
        "account_id", "debit", "credit", "balance",
    ], limit=0)
    if not lines_data:
        return "No journal entries found for {} to {} (company={}).".format(date_from, date_to, company or "all")

    account_ids = list(set(l["account_id"][0] for l in lines_data if l.get("account_id")))
    accounts = odoo.search_read("account.account", [("id", "in", account_ids)], [
        "name", "code", "account_type",
    ])
    acc_map = {a["id"]: a for a in accounts}

    agg = defaultdict(lambda: {"debit": 0, "credit": 0, "balance": 0})
    for line in lines_data:
        acc_id = line["account_id"][0] if line.get("account_id") else None
        if not acc_id or acc_id not in acc_map:
            continue
        label = "{} {}".format(acc_map[acc_id].get("code", ""), acc_map[acc_id].get("name", ""))
        agg[label]["debit"] += line.get("debit", 0) or 0
        agg[label]["credit"] += line.get("credit", 0) or 0
        agg[label]["balance"] += line.get("balance", 0) or 0

    total_debit = sum(v["debit"] for v in agg.values())
    total_credit = sum(v["credit"] for v in agg.values())

    out = ["TRIAL BALANCE: {} to {} (company: {})".format(date_from, date_to, company or "all")]
    out.append("=" * 70)
    out.append("  {:40s} {:>12s} {:>12s} {:>12s}".format("Account", "Debit", "Credit", "Balance"))
    out.append("  " + "-" * 66)
    for label, v in sorted(agg.items()):
        if abs(v["debit"]) > 0.01 or abs(v["credit"]) > 0.01:
            out.append("  {:40s} {:>12,.2f} {:>12,.2f} {:>12,.2f}".format(
                label[:40], v["debit"], v["credit"], v["balance"]))
    out.append("  " + "-" * 66)
    out.append("  {:40s} {:>12,.2f} {:>12,.2f}".format("TOTALS", total_debit, total_credit))
    return "\n".join(out)


@tool
def query_general_ledger(account_code: str = "", date_from: str = "",
                         date_to: str = "", company: str = "", limit: int = 200) -> str:
    """General Ledger: detailed journal entries for a specific account code.
    Shows every posted entry with date, journal, partner, ref, debit, credit, balance.
    account_code: e.g. '400000' (Sales Revenue), '500000' (Expenses).
    date_from/date_to: YYYY-MM-DD (required - user must provide)."""
    if not account_code:
        return "Please provide an account_code (e.g. '400000')."
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("date", ">=", date_from),
        ("date", "<=", date_to),
        ("account_id.code", "=like", account_code + "%"),
    ] + _cd(company)

    entries = odoo.search_read("account.move.line", domain, [
        "date", "move_id", "partner_id", "ref", "name",
        "debit", "credit", "balance", "account_id",
    ], limit=limit, order="date asc, id asc")
    if not entries:
        return "No entries for account '{}' from {} to {} (company={}).".format(
            account_code, date_from, date_to, company or "all")

    total_debit = sum(e.get("debit", 0) or 0 for e in entries)
    total_credit = sum(e.get("credit", 0) or 0 for e in entries)
    running = 0

    out = ["GENERAL LEDGER: Account {} | {} to {} (company: {})".format(
        account_code, date_from, date_to, company or "all")]
    out.append("  {} entries found (limit={})".format(len(entries), limit))
    out.append("")
    for e in entries[:100]:
        running += (e.get("debit", 0) or 0) - (e.get("credit", 0) or 0)
        out.append("  {} | {} | {} | D:{:,.2f} C:{:,.2f} | Bal:{:,.2f}".format(
            _s(e.get("date"), ""),
            _rn(e.get("move_id"))[:20],
            _rn(e.get("partner_id"))[:25],
            e.get("debit", 0) or 0,
            e.get("credit", 0) or 0,
            running))
    if len(entries) > 100:
        out.append("  ... {} more entries".format(len(entries) - 100))
    out.append("")
    out.append("  TOTALS: Debit={:,.2f}  Credit={:,.2f}  Net={:,.2f}".format(
        total_debit, total_credit, total_debit - total_credit))
    return "\n".join(out)


@tool
def query_aged_receivable(as_of: str = "", company: str = "") -> str:
    """Aged Receivable report. Shows how much each customer owes grouped by aging buckets
    (Current, 1-30, 31-60, 61-90, 90+ days). Based on open receivable balances.
    as_of: YYYY-MM-DD (required - user must provide)."""
    if not as_of:
        return "Please provide as_of date (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("account_id.account_type", "=", "asset_receivable"),
        ("amount_residual", "!=", 0),
    ] + _cd(company)

    lines_data = odoo.search_read("account.move.line", domain, [
        "partner_id", "date_maturity", "date", "amount_residual",
    ], limit=0)
    if not lines_data:
        return "No open receivables found (company={}).".format(company or "all")

    try:
        ref_date = datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        ref_date = datetime.now()

    buckets = {"Current": 0, "1-30": 0, "31-60": 0, "61-90": 0, "90+": 0}
    by_partner = defaultdict(lambda: {"total": 0, "Current": 0, "1-30": 0, "31-60": 0, "61-90": 0, "90+": 0})

    for line in lines_data:
        amt = line.get("amount_residual", 0) or 0
        due = _s(line.get("date_maturity"), _s(line.get("date"), as_of))
        try:
            days = (ref_date - datetime.strptime(str(due)[:10], "%Y-%m-%d")).days
        except (ValueError, TypeError):
            days = 0
        if days <= 0:
            bucket = "Current"
        elif days <= 30:
            bucket = "1-30"
        elif days <= 60:
            bucket = "31-60"
        elif days <= 90:
            bucket = "61-90"
        else:
            bucket = "90+"
        buckets[bucket] += amt
        partner = _rn(line.get("partner_id"))
        by_partner[partner]["total"] += amt
        by_partner[partner][bucket] += amt

    total = sum(buckets.values())
    out = ["AGED RECEIVABLE as of {} (company: {})".format(as_of, company or "all")]
    out.append("=" * 55)
    out.append("  Total Outstanding: {:,.2f}".format(total))
    out.append("")
    for b in ["Current", "1-30", "31-60", "61-90", "90+"]:
        out.append("    {} days: {:,.2f}".format(b, buckets[b]))
    out.append("")
    top = sorted(by_partner.items(), key=lambda x: -abs(x[1]["total"]))[:20]
    out.append("  Top 20 Debtors:")
    for name, data in top:
        out.append("    {}: {:,.2f} (90+: {:,.2f})".format(name[:35], data["total"], data["90+"]))
    return "\n".join(out)


@tool
def query_aged_payable(as_of: str = "", company: str = "") -> str:
    """Aged Payable report. Shows how much is owed to each vendor by aging buckets.
    as_of: YYYY-MM-DD (required - user must provide)."""
    if not as_of:
        return "Please provide as_of date (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("account_id.account_type", "=", "liability_payable"),
        ("amount_residual", "!=", 0),
    ] + _cd(company)

    lines_data = odoo.search_read("account.move.line", domain, [
        "partner_id", "date_maturity", "date", "amount_residual",
    ], limit=0)
    if not lines_data:
        return "No open payables found (company={}).".format(company or "all")

    try:
        ref_date = datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        ref_date = datetime.now()

    buckets = {"Current": 0, "1-30": 0, "31-60": 0, "61-90": 0, "90+": 0}
    by_partner = defaultdict(lambda: {"total": 0})

    for line in lines_data:
        amt = abs(line.get("amount_residual", 0) or 0)
        due = _s(line.get("date_maturity"), _s(line.get("date"), as_of))
        try:
            days = (ref_date - datetime.strptime(str(due)[:10], "%Y-%m-%d")).days
        except (ValueError, TypeError):
            days = 0
        bucket = "Current" if days <= 0 else "1-30" if days <= 30 else "31-60" if days <= 60 else "61-90" if days <= 90 else "90+"
        buckets[bucket] += amt
        by_partner[_rn(line.get("partner_id"))]["total"] += amt

    total = sum(buckets.values())
    out = ["AGED PAYABLE as of {} (company: {})".format(as_of, company or "all")]
    out.append("=" * 55)
    out.append("  Total Payable: {:,.2f}".format(total))
    for b in ["Current", "1-30", "31-60", "61-90", "90+"]:
        out.append("    {} days: {:,.2f}".format(b, buckets[b]))
    out.append("")
    top = sorted(by_partner.items(), key=lambda x: -x[1]["total"])[:20]
    out.append("  Top 20 Creditors:")
    for name, data in top:
        out.append("    {}: {:,.2f}".format(name[:35], data["total"]))
    return "\n".join(out)


@tool
def query_partner_ledger(partner_name: str = "", date_from: str = "",
                         date_to: str = "", company: str = "", limit: int = 200) -> str:
    """Partner Ledger: all journal entries for a specific customer/vendor.
    partner_name: name or partial name to search for.
    date_from/date_to: YYYY-MM-DD (required - user must provide)."""
    if not partner_name:
        return "Please provide a partner_name to search for."
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD)."
    partners = odoo.search_read("res.partner", [("name", "ilike", partner_name)],
        ["name"], limit=5)
    if not partners:
        return "No partner found matching '{}'.".format(partner_name)

    partner_ids = [p["id"] for p in partners]
    domain = [
        ("move_id.state", "=", "posted"),
        ("partner_id", "in", partner_ids),
        ("date", ">=", date_from),
        ("date", "<=", date_to),
    ] + _cd(company)

    entries = odoo.search_read("account.move.line", domain, [
        "date", "move_id", "account_id", "name", "ref",
        "debit", "credit", "balance", "amount_residual",
    ], limit=limit, order="date asc")
    if not entries:
        return "No entries for partner '{}' from {} to {}.".format(partner_name, date_from, date_to)

    total_debit = sum(e.get("debit", 0) or 0 for e in entries)
    total_credit = sum(e.get("credit", 0) or 0 for e in entries)
    total_residual = sum(e.get("amount_residual", 0) or 0 for e in entries)

    out = ["PARTNER LEDGER: '{}' ({} entries, {} to {})".format(
        partner_name, len(entries), date_from, date_to)]
    out.append("  Partners matched: {}".format(", ".join(p["name"] for p in partners)))
    out.append("")
    for e in entries[:80]:
        out.append("  {} | {} | {} | D:{:,.2f} C:{:,.2f} | Res:{:,.2f}".format(
            _s(e.get("date"), ""),
            _rn(e.get("move_id"))[:20],
            _rn(e.get("account_id"))[:25],
            e.get("debit", 0) or 0,
            e.get("credit", 0) or 0,
            e.get("amount_residual", 0) or 0))
    if len(entries) > 80:
        out.append("  ... {} more entries".format(len(entries) - 80))
    out.append("")
    out.append("  TOTALS: Debit={:,.2f}  Credit={:,.2f}  Open Balance={:,.2f}".format(
        total_debit, total_credit, total_residual))
    return "\n".join(out)


@tool
def query_cash_flow(date_from: str = "", date_to: str = "",
                    company: str = "") -> str:
    """Cash Flow report from bank/cash journal entries.
    Shows cash inflows and outflows by partner/source.
    date_from/date_to: YYYY-MM-DD (required - user must provide)."""
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD)."
    domain = [
        ("move_id.state", "=", "posted"),
        ("date", ">=", date_from),
        ("date", "<=", date_to),
        ("account_id.account_type", "=", "asset_cash"),
    ] + _cd(company)

    entries = odoo.search_read("account.move.line", domain, [
        "date", "partner_id", "debit", "credit", "name", "ref", "journal_id",
    ], limit=0)
    if not entries:
        return "No cash entries found for {} to {} (company={}).".format(date_from, date_to, company or "all")

    total_in = sum(e.get("debit", 0) or 0 for e in entries)
    total_out = sum(e.get("credit", 0) or 0 for e in entries)
    net = total_in - total_out

    by_journal = defaultdict(lambda: {"in": 0, "out": 0})
    by_partner = defaultdict(lambda: {"in": 0, "out": 0})
    for e in entries:
        j = _rn(e.get("journal_id"))
        p = _rn(e.get("partner_id"))
        by_journal[j]["in"] += e.get("debit", 0) or 0
        by_journal[j]["out"] += e.get("credit", 0) or 0
        by_partner[p]["in"] += e.get("debit", 0) or 0
        by_partner[p]["out"] += e.get("credit", 0) or 0

    out = ["CASH FLOW: {} to {} (company: {})".format(date_from, date_to, company or "all")]
    out.append("=" * 55)
    out.append("  Cash In:  {:,.2f}".format(total_in))
    out.append("  Cash Out: {:,.2f}".format(total_out))
    out.append("  Net:      {:,.2f}".format(net))
    out.append("")
    out.append("  By Journal:")
    for j, v in sorted(by_journal.items(), key=lambda x: -(x[1]["in"] + x[1]["out"])):
        out.append("    {}: In={:,.2f} Out={:,.2f} Net={:,.2f}".format(j, v["in"], v["out"], v["in"]-v["out"]))
    out.append("")
    out.append("  Top 15 by Volume:")
    top = sorted(by_partner.items(), key=lambda x: -(x[1]["in"] + x[1]["out"]))[:15]
    for p, v in top:
        out.append("    {}: In={:,.2f} Out={:,.2f}".format(p[:35], v["in"], v["out"]))
    return "\n".join(out)


@tool
def search_anything(query: str = "", models_csv: str = "") -> str:
    """Universal search across the entire business database.
    Searches by name/reference in multiple models simultaneously.
    query: text to search for (name, reference, employee name, etc.)
    models_csv: optional comma-separated models to search. If empty, searches
    common models: employees, sales, operations, partners, invoices, tickets, etc."""
    if not query:
        return "Please provide a search query."
    if models_csv:
        models_to_search = [m.strip() for m in models_csv.split(",")]
    else:
        models_to_search = [
            "hr.employee", "res.partner", "sale.order", "account.move",
            "project.operation", "crm.lead", "ticket.helpdesk",
            "employee.kpi", "hr.payslip", "cus.purchase.requisition",
            "bp.tender", "profile.db", "product.lab.project",
        ]
    results = []
    for model in models_to_search:
        try:
            # Search by name field (works for most models)
            recs = odoo.search_read(model, [("name", "ilike", query)],
                ["name"], limit=10, order="create_date desc")
            if recs:
                results.append("  {} ({} hits):".format(model, len(recs)))
                for r in recs[:5]:
                    results.append("    [{}] {}".format(r["id"], _s(r.get("name"), "-")))
                if len(recs) > 5:
                    results.append("    ... {} more".format(len(recs) - 5))
        except Exception:
            pass
    if not results:
        return "No results for '{}' across {} models.".format(query, len(models_to_search))
    out = ["Search results for '{}':".format(query)]
    out.extend(results)
    return "\n".join(out)


@tool
def lookup_employee_kpi(employee_name: str = "", date_from: str = "", date_to: str = "",
                        quarter: str = "", role_type: str = "", company: str = "") -> str:
    """Lookup individual employee KPI + bonus details by name.
    Shows all KPI records for that employee with targets, sales, paid, unpaid, carry, shortfall, bonus.
    date_from / date_to: filter by created date range (YYYY-MM-DD). Example: date_from=2025-10-01, date_to=2026-02-28.
    quarter: optional filter q1|q2|q3|q4 (q1=Jul-Sep, q2=Oct-Dec, q3=Jan-Mar, q4=Apr-Jun).
    role_type: optional filter 'sale' or 'operation'."""
    if not employee_name:
        return "Please provide an employee name."
    if not date_from or not date_to:
        return "Please provide both date_from and date_to (YYYY-MM-DD) to define the KPI lookup period."
    domain = [("employee_id.name", "ilike", employee_name)]
    if date_from:
        domain.append(("create_date", ">=", date_from))
    if date_to:
        domain.append(("create_date", "<=", date_to + " 23:59:59"))
    if quarter:
        domain.append(("quarter", "=", quarter))
    if role_type:
        domain.append(("role_type", "=", role_type))
    domain += _cd(company)

    kpis = odoo.search_read("employee.kpi", domain, [
        "employee_id", "create_date", "year", "quarter", "role_type", "grade_id",
        "minimum_target", "total_sales", "total_paid", "total_unpaid",
        "bonus_amount", "eligible_bonus", "carry_amount", "carry_paid_sales",
        "shortfall_amount", "is_penalty", "state", "period_start", "period_end",
    ], limit=0, order="create_date desc")
    if not kpis:
        return "No KPI records for employee '{}' between {} and {}.".format(
            employee_name, date_from or "?", date_to or "?")

    total_sales = sum(k.get("total_sales", 0) or 0 for k in kpis)
    total_paid = sum(k.get("total_paid", 0) or 0 for k in kpis)
    total_unpaid = sum(k.get("total_unpaid", 0) or 0 for k in kpis)
    total_bonus = sum(k.get("bonus_amount", 0) or 0 for k in kpis)
    total_shortfall = sum(k.get("shortfall_amount", 0) or 0 for k in kpis)

    out = ["KPI for '{}' ({} records, {} to {}):".format(
        employee_name, len(kpis), date_from, date_to)]
    out.append("  Totals => Sales: ${:,.2f} | Paid: ${:,.2f} | Unpaid: ${:,.2f} | Bonus: {:,.2f} BDT | Shortfall: ${:,.2f}".format(
        total_sales, total_paid, total_unpaid, total_bonus, total_shortfall))
    out.append("")
    for k in kpis:
        created = _s(k.get("create_date"), "?")
        if created and len(created) > 10:
            created = created[:10]
        out.append("  Record (Created: {}) — {} Q{}:".format(
            created, _s(k.get("year"), "?"), _s(k.get("quarter"), "?")))
        out.append("    Role: {} | Grade: {} | State: {}{}".format(
            _s(k.get("role_type"), "-"), _rn(k.get("grade_id")),
            _s(k.get("state"), "?"),
            " [PENALTY]" if k.get("is_penalty") else ""))
        out.append("    Period: {} to {}".format(
            _s(k.get("period_start"), "?"), _s(k.get("period_end"), "?")))
        out.append("    Min Target: ${:,.2f} | Total Sales: ${:,.2f}".format(
            k.get("minimum_target", 0) or 0, k.get("total_sales", 0) or 0))
        out.append("    Paid: ${:,.2f} | Unpaid: ${:,.2f}".format(
            k.get("total_paid", 0) or 0, k.get("total_unpaid", 0) or 0))
        out.append("    New Carry: {:,.2f} BDT | Carry Sales: ${:,.2f}".format(
            k.get("carry_amount", 0) or 0, k.get("carry_paid_sales", 0) or 0))
        out.append("    Eligible Bonus: {:,.2f} BDT | Bonus Amount: {:,.2f} BDT".format(
            k.get("eligible_bonus", 0) or 0, k.get("bonus_amount", 0) or 0))
        out.append("    Shortfall: ${:,.2f}".format(k.get("shortfall_amount", 0) or 0))
        out.append("")
    return "\n".join(out)


@tool
def lookup_employee_detail(employee_name: str = "") -> str:
    """Lookup detailed employee information by name.
    Shows department, job, company, contract, attendance summary, loans."""
    if not employee_name:
        return "Please provide an employee name."
    emps = odoo.search_read("hr.employee", [("name", "ilike", employee_name)], [
        "name", "department_id", "job_id", "job_title", "company_id",
        "work_email", "work_phone", "parent_id", "coach_id",
    ], limit=5)
    if not emps:
        return "No employee found matching '{}'.".format(employee_name)

    out = []
    for emp in emps:
        out.append("Employee: {} (ID: {})".format(emp["name"], emp["id"]))
        out.append("  Department: {}".format(_rn(emp.get("department_id"))))
        out.append("  Job: {} / {}".format(_rn(emp.get("job_id")), _s(emp.get("job_title"), "-")))
        out.append("  Company: {}".format(_rn(emp.get("company_id"))))
        out.append("  Email: {}  Phone: {}".format(_s(emp.get("work_email"), "-"), _s(emp.get("work_phone"), "-")))
        out.append("  Manager: {}  Coach: {}".format(_rn(emp.get("parent_id")), _rn(emp.get("coach_id"))))

        # Contract
        contracts = odoo.search_read("hr.contract", [
            ("employee_id", "=", emp["id"]), ("state", "=", "open")
        ], ["name", "wage", "date_start", "structure_type_id"], limit=1)
        if contracts:
            c = contracts[0]
            out.append("  Contract: {} | Wage: {:,.0f} BDT | Since: {}".format(
                c.get("name", ""), c.get("wage", 0) or 0, _s(c.get("date_start"), "-")))

        # Loans
        loans = odoo.search_read("hr.loan", [("employee_id", "=", emp["id"])],
            ["loan_amount", "balance_amount", "state"], limit=5)
        if loans:
            for l in loans:
                out.append("  Loan: {:,.0f} BDT (balance: {:,.0f}) state={}".format(
                    l.get("loan_amount", 0) or 0, l.get("balance_amount", 0) or 0, _s(l.get("state"), "?")))
        out.append("")
    return "\n".join(out)


@tool
def query_team_performance(team_name: str = "", date_from: str = "", date_to: str = "",
                           company: str = "") -> str:
    """Query sales/operations performance by team. Shows revenue, delivery rate, top members.
    team_name: search by team name (partial match). Leave empty for all teams."""
    # Sales by team
    so_domain = _cd(company)
    if date_from:
        so_domain.append(("date_order", ">=", date_from))
    if date_to:
        so_domain.append(("date_order", "<=", date_to))

    orders = odoo.search_read("sale.order", so_domain, [
        "team_id", "user_id", "amount_total", "state",
    ], limit=0)

    by_team = defaultdict(lambda: {"count": 0, "revenue": 0, "members": defaultdict(float)})
    for o in orders:
        t = _rn(o.get("team_id"))
        if team_name and team_name.lower() not in t.lower():
            continue
        amt = o.get("amount_total", 0) or 0
        by_team[t]["count"] += 1
        by_team[t]["revenue"] += amt
        by_team[t]["members"][_rn(o.get("user_id"))] += amt

    if not by_team:
        return "No sales data found for team '{}'.".format(team_name or "all")

    out = ["TEAM PERFORMANCE (company: {}):".format(company or "all")]
    for team, data in sorted(by_team.items(), key=lambda x: -x[1]["revenue"]):
        out.append("")
        out.append("  Team: {}".format(team))
        out.append("    Orders: {}  Revenue: ${:,.2f}".format(data["count"], data["revenue"]))
        out.append("    Members:")
        for member, rev in sorted(data["members"].items(), key=lambda x: -x[1])[:10]:
            out.append("      {}: ${:,.2f}".format(member, rev))
    return "\n".join(out)


@tool
def query_service_line_analysis(company: str = "") -> str:
    """Analyze operations by service line/type. Shows revenue, count, delivery rate per service."""
    ops = odoo.search_read("project.operation", _cd(company), [
        "service_type_id", "order_status", "delivery_amount", "monetary_value",
    ], limit=0)
    if not ops:
        return "No operations data found."

    by_service = defaultdict(lambda: {"count": 0, "delivered": 0, "value": 0})
    for o in ops:
        svc = _rn(o.get("service_type_id"))
        by_service[svc]["count"] += 1
        st = _s(o.get("order_status"), "")
        if st in ("delivered", "complete"):
            by_service[svc]["delivered"] += 1
        by_service[svc]["value"] += o.get("delivery_amount", 0) or o.get("monetary_value", 0) or 0

    out = ["SERVICE LINE ANALYSIS (company: {}):".format(company or "all")]
    for svc, data in sorted(by_service.items(), key=lambda x: -x[1]["value"]):
        rate = (data["delivered"] / data["count"] * 100) if data["count"] else 0
        out.append("  {}: {} ops | ${:,.2f} | {:.0f}% delivered".format(
            svc, data["count"], data["value"], rate))
    return "\n".join(out)    systemctl restart odoo


@tool
def query_budget(year: str = "2025", company: str = "") -> str:
    """Query crossovered.budget (budget vs actual)."""
    domain = [("date_from", ">=", "{}-01-01".format(year)), ("date_to", "<=", "{}-12-31".format(year))]
    domain += _cd(company)
    budgets = odoo.search_read("crossovered.budget", domain,
        ["name", "state", "date_from", "date_to", "crossovered_budget_line"])
    if not budgets:
        return "No budgets for {}.".format(year)

    line_ids = []
    for b in budgets:
        line_ids.extend(b.get("crossovered_budget_line", []))
    if not line_ids:
        return "Budgets exist for {} but no lines.".format(year)

    bl = odoo.search_read("crossovered.budget.lines", [("id", "in", line_ids)],
        ["general_budget_id", "planned_amount", "practical_amount"])
    planned = sum(l.get("planned_amount", 0) or 0 for l in bl)
    actual = sum(l.get("practical_amount", 0) or 0 for l in bl)
    return ("Budget {} ({} lines):\n"
            "  Planned: ${:,.2f}\n  Actual: ${:,.2f}\n"
            "  Variance: ${:,.2f} ({})").format(
        year, len(bl), abs(planned), abs(actual),
        actual - planned, "over" if actual > planned else "under")


# ========================================================================
# 9. TOOLS - PAYROLL (limit=0)
# ========================================================================

@tool
def query_payroll(month: str = "", year: str = "2025", limit: int = 0, company: str = "") -> str:
    """Query hr.payslip records for salary analysis. limit=0 = ALL records.
    month: '01'-'12' or '' for all. year: e.g. '2025'."""
    domain = _cd(company)
    if year and month:
        domain += [("date_from", ">=", "{}-{}-01".format(year, month)),
                   ("date_to", "<=", "{}-{}-28".format(year, month))]
    elif year:
        domain += [("date_from", ">=", "{}-01-01".format(year)), ("date_to", "<=", "{}-12-31".format(year))]

    slips = odoo.search_read("hr.payslip", domain, [
        "employee_id", "name", "state", "net_wage", "basic_wage",
        "date_from", "date_to", "company_id", "struct_id",
    ], limit=limit, order="date_from desc")
    if not slips:
        return "No payslips found."

    total_net = sum(s.get("net_wage", 0) or 0 for s in slips)
    total_basic = sum(s.get("basic_wage", 0) or 0 for s in slips)
    by_state = defaultdict(int)
    by_company = defaultdict(float)
    for s in slips:
        by_state[_s(s.get("state"), "unknown")] += 1
        by_company[_rn(s.get("company_id"))] += s.get("net_wage", 0) or 0

    lines = ["Payroll ({} payslips):".format(len(slips)),
             "  Total Basic: {:,.0f} BDT".format(total_basic),
             "  Total Net: {:,.0f} BDT".format(total_net),
             "  By State:"]
    for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    if len(by_company) > 1:
        lines.append("  By Company:")
        for co, amt in sorted(by_company.items(), key=lambda x: -x[1]):
            lines.append("    {}: {:,.0f} BDT".format(co, amt))
    return "\n".join(lines)


@tool
def query_loans(status: str = "all", company: str = "") -> str:
    """Query hr.loan records (employee loans). status: all|draft|approve|paid|refuse."""
    domain = _cd(company)
    if status != "all":
        domain.append(("state", "=", status))
    loans = odoo.search_read("hr.loan", domain, [
        "employee_id", "loan_amount", "total_amount", "balance_amount",
        "state", "date", "company_id",
    ], limit=0, order="date desc")
    if not loans:
        return "No loans found."
    total = sum(l.get("loan_amount", 0) or 0 for l in loans)
    outstanding = sum(l.get("balance_amount", 0) or 0 for l in loans)
    by_state = defaultdict(int)
    for l in loans:
        by_state[_s(l.get("state"), "unknown")] += 1
    lines = ["Loans ({} records):".format(len(loans)),
             "  Total Disbursed: {:,.0f} BDT".format(total),
             "  Outstanding: {:,.0f} BDT".format(outstanding),
             "  By State:"]
    for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


# ========================================================================
# 10. TOOLS - ATTENDANCE
# ========================================================================

@tool
def query_attendance(date_from: str = "", date_to: str = "", limit: int = 0, company: str = "") -> str:
    """Query employee.attendance.details for attendance tracking. limit=0 = ALL.
    Dates: YYYY-MM-DD."""
    domain = _cd(company)
    if date_from:
        domain.append(("date", ">=", date_from))
    if date_to:
        domain.append(("date", "<=", date_to))

    recs = odoo.search_read("employee.attendance.details", domain, [
        "employee_id", "date", "status", "check_in", "check_out",
        "working_hours", "late_minutes", "company_id",
    ], limit=limit, order="date desc")
    if not recs:
        return "No attendance records found."

    by_status = defaultdict(int)
    total_late = 0
    for r in recs:
        by_status[_s(r.get("status"), "unknown")] += 1
        total_late += r.get("late_minutes", 0) or 0

    lines = ["Attendance ({} records):".format(len(recs)),
             "  Total Late Minutes: {}".format(total_late),
             "  By Status:"]
    for st, c in sorted(by_status.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


@tool
def query_late_penalties(year: str = "2025", company: str = "") -> str:
    """Query hr.late.penalty records for employee discipline tracking."""
    domain = _cd(company)
    if year:
        domain += [("date", ">=", "{}-01-01".format(year)), ("date", "<=", "{}-12-31".format(year))]
    recs = odoo.search_read("hr.late.penalty", domain, [
        "employee_id", "date", "penalty_amount", "state",
    ], limit=0)
    if not recs:
        return "No late penalties found."
    total = sum(r.get("penalty_amount", 0) or 0 for r in recs)
    lines = ["Late Penalties ({} records, {}):".format(len(recs), year),
             "  Total Penalty Amount: {:,.0f} BDT".format(total)]
    return "\n".join(lines)


# ========================================================================
# 11. TOOLS - HR
# ========================================================================

@tool
def query_employees(department: str = "", limit: int = 0, company: str = "") -> str:
    """Query hr.employee. Filter by department name. limit=0 = ALL."""
    domain = [("active", "=", True)] + _cd(company)
    if department:
        domain.append(("department_id.name", "ilike", department))
    emps = odoo.search_read("hr.employee", domain, [
        "name", "department_id", "job_id", "job_title", "company_id",
    ], limit=limit)
    if not emps:
        return "No employees found."

    by_dept = defaultdict(int)
    by_job = defaultdict(int)
    by_company = defaultdict(int)
    for e in emps:
        by_dept[_rn(e.get("department_id"))] += 1
        by_job[_rn(e.get("job_id"))] += 1
        by_company[_rn(e.get("company_id"))] += 1

    lines = ["Employees ({} active):".format(len(emps)),
             "  By Department:"]
    for d, c in sorted(by_dept.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(d, c))
    lines.append("  By Job (top 10):")
    for j, c in sorted(by_job.items(), key=lambda x: -x[1])[:10]:
        lines.append("    {}: {}".format(j, c))
    if len(by_company) > 1:
        lines.append("  By Company:")
        for co, c in sorted(by_company.items(), key=lambda x: -x[1]):
            lines.append("    {}: {}".format(co, c))
    return "\n".join(lines)


# ========================================================================
# 12. TOOLS - PROCUREMENT & PURCHASE
# ========================================================================

@tool
def query_procurement(status: str = "all", limit: int = 0, company: str = "") -> str:
    """Query bp.tender (procurement tenders) and bp.bid (vendor bids).
    Shows tender pipeline, bid status, and vendor participation."""
    td = _cd(company)
    if status != "all":
        td.append(("state", "=", status))
    tenders = odoo.search_read("bp.tender", td, [
        "name", "state", "deadline", "category_id", "budget_estimate",
    ], limit=limit, order="create_date desc")

    bids = odoo.search_read("bp.bid", _cd(company), [
        "tender_id", "vendor_id", "state", "total_amount",
    ], limit=limit)

    lines = ["Procurement Overview:"]
    if tenders:
        by_state = defaultdict(int)
        total_budget = 0
        for t in tenders:
            by_state[_s(t.get("state"), "unknown")] += 1
            total_budget += t.get("budget_estimate", 0) or 0
        lines.append("  Tenders ({}):".format(len(tenders)))
        lines.append("    Est. Budget: ${:,.2f}".format(total_budget))
        for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
            lines.append("    {}: {}".format(st, c))
    else:
        lines.append("  Tenders: none found")

    if bids:
        bid_total = sum(b.get("total_amount", 0) or 0 for b in bids)
        bid_states = defaultdict(int)
        for b in bids:
            bid_states[_s(b.get("state"), "unknown")] += 1
        lines.append("  Bids ({}):".format(len(bids)))
        lines.append("    Total Bid Value: ${:,.2f}".format(bid_total))
        for st, c in sorted(bid_states.items(), key=lambda x: -x[1]):
            lines.append("    {}: {}".format(st, c))
    else:
        lines.append("  Bids: none found")
    return "\n".join(lines)


@tool
def query_purchase_requisitions(status: str = "all", company: str = "") -> str:
    """Query cus.purchase.requisition (internal purchase approval requests)."""
    domain = _cd(company)
    if status != "all":
        domain.append(("state", "=", status))
    reqs = odoo.search_read("cus.purchase.requisition", domain, [
        "name", "employee_id", "department_id", "state",
        "total_amount", "date", "company_id",
    ], limit=0, order="date desc")
    if not reqs:
        return "No purchase requisitions found."

    total = sum(r.get("total_amount", 0) or 0 for r in reqs)
    by_state = defaultdict(int)
    for r in reqs:
        by_state[_s(r.get("state"), "unknown")] += 1
    lines = ["Purchase Requisitions ({}):".format(len(reqs)),
             "  Total Amount: ${:,.2f}".format(total),
             "  By State:"]
    for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


# ========================================================================
# 13. TOOLS - HELPDESK
# ========================================================================

@tool
def query_helpdesk(status: str = "all", limit: int = 0, company: str = "") -> str:
    """Query ticket.helpdesk (support tickets). Shows ticket volume, categories, stages."""
    domain = _cd(company)
    if status != "all":
        domain.append(("stage_id.name", "ilike", status))
    tickets = odoo.search_read("ticket.helpdesk", domain, [
        "name", "partner_id", "stage_id", "category_id",
        "team_id", "user_id", "priority",
    ], limit=limit)
    if not tickets:
        return "No helpdesk tickets found."

    by_stage = defaultdict(int)
    by_cat = defaultdict(int)
    for t in tickets:
        by_stage[_rn(t.get("stage_id"))] += 1
        by_cat[_rn(t.get("category_id"))] += 1
    lines = ["Helpdesk ({} tickets):".format(len(tickets)),
             "  By Stage:"]
    for st, c in sorted(by_stage.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    lines.append("  By Category:")
    for cat, c in sorted(by_cat.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(cat, c))
    return "\n".join(lines)


# ========================================================================
# 14. TOOLS - PROFILE DATABASE
# ========================================================================

@tool
def query_profile_database(limit: int = 0, company: str = "") -> str:
    """Query profile.db (freelancer marketplace profiles - Fiverr, Upwork, etc.)."""
    domain = _cd(company)
    profiles = odoo.search_read("profile.db", domain, [
        "name", "employee_id", "platform", "status", "company_id",
    ], limit=limit)
    if not profiles:
        return "No profiles found in profile.db."

    by_platform = defaultdict(int)
    by_status = defaultdict(int)
    for p in profiles:
        by_platform[_s(p.get("platform"), "unknown")] += 1
        by_status[_s(p.get("status"), "unknown")] += 1
    lines = ["Profile Database ({} profiles):".format(len(profiles)),
             "  By Platform:"]
    for pl, c in sorted(by_platform.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(pl, c))
    lines.append("  By Status:")
    for st, c in sorted(by_status.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


# ========================================================================
# 15. TOOLS - ASSETS
# ========================================================================

@tool
def query_assets(status: str = "all", company: str = "") -> str:
    """Query account.asset.asset (fixed assets - computers, furniture, etc.)."""
    domain = _cd(company)
    if status != "all":
        domain.append(("state", "=", status))
    assets = odoo.search_read("account.asset.asset", domain, [
        "name", "category_id", "value", "value_residual", "state",
        "date", "company_id",
    ], limit=0)
    if not assets:
        return "No assets found."

    total_val = sum(a.get("value", 0) or 0 for a in assets)
    total_res = sum(a.get("value_residual", 0) or 0 for a in assets)
    by_state = defaultdict(int)
    for a in assets:
        by_state[_s(a.get("state"), "unknown")] += 1
    lines = ["Assets ({}):".format(len(assets)),
             "  Total Value: ${:,.2f}".format(total_val),
             "  Residual: ${:,.2f}".format(total_res),
             "  By State:"]
    for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


# ========================================================================
# 16. TOOLS - PRODUCT LAB
# ========================================================================

@tool
def query_product_lab(limit: int = 0) -> str:
    """Query product.lab.project (R&D / product development projects)."""
    projects = odoo.search_read("product.lab.project", [], [
        "name", "industry_id", "partner_id", "state",
    ], limit=limit)
    if not projects:
        return "No product lab projects found."
    by_state = defaultdict(int)
    for p in projects:
        by_state[_s(p.get("state"), "unknown")] += 1
    lines = ["Product Lab ({} projects):".format(len(projects)),
             "  By State:"]
    for st, c in sorted(by_state.items(), key=lambda x: -x[1]):
        lines.append("    {}: {}".format(st, c))
    return "\n".join(lines)


# ========================================================================
# 17. TOOLS - ANALYTICAL
# ========================================================================

@tool
def get_exchange_rates() -> str:
    """Get BDT/USD exchange rates from Odoo."""
    curs = odoo.search_read("res.currency", [("name", "in", ["USD", "BDT"])],
        ["name", "rate", "symbol"])
    if not curs:
        return "Currency data unavailable."
    rates = {c["name"]: c.get("rate", 1) for c in curs}
    usd = rates.get("USD", 1)
    bdt = rates.get("BDT", 120)
    cross = bdt / usd if usd else 120
    return ("Exchange Rates:\n  USD rate: {}\n  BDT rate: {}\n"
            "  BDT per USD: {:.2f}\n  USD per BDT: {:.6f}").format(usd, bdt, cross, 1.0/cross if cross else 0)


@tool
def calculate_solvency_score(company: str = "") -> str:
    """Altman Z''-Score (private firms). Call AFTER reviewing accounting data."""
    cd = _cd(company)
    inv = odoo.search_read("account.move",
        [("state", "=", "posted"), ("move_type", "=", "out_invoice")] + cd,
        ["amount_total", "amount_residual"], limit=0)
    bil = odoo.search_read("account.move",
        [("state", "=", "posted"), ("move_type", "=", "in_invoice")] + cd,
        ["amount_total", "amount_residual"], limit=0)

    revenue = sum(i.get("amount_total", 0) or 0 for i in inv)
    ar = sum(i.get("amount_residual", 0) or 0 for i in inv)
    expenses = sum(b.get("amount_total", 0) or 0 for b in bil)
    ap = sum(b.get("amount_residual", 0) or 0 for b in bil)

    if revenue == 0 and expenses == 0:
        return "Insufficient accounting data for Z-Score calculation."

    cash = max(revenue - expenses, 0) * 0.15
    ta = max(revenue * 0.6, 1)
    tl = max(expenses * 0.4, 1)
    eq = ta - tl
    re = eq * 0.3
    ebit = revenue - expenses

    x1 = (cash + ar - ap) / ta
    x2 = re / ta
    x3 = ebit / ta
    x4 = eq / max(tl, 1)
    z = round(6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4, 2)

    zone = "SAFE" if z >= 2.99 else "GREY" if z >= 1.81 else "DISTRESS"
    return ("Z-Score: {} ({})\n  X1={:.3f} X2={:.3f} X3={:.3f} X4={:.3f}\n"
            "  Revenue=${:,.0f} Expenses=${:,.0f} AR=${:,.0f} AP=${:,.0f}").format(
        z, zone, x1, x2, x3, x4, revenue, expenses, ar, ap)


@tool
def run_monte_carlo_runway(company: str = "") -> str:
    """5000-iteration Monte Carlo cash runway simulation."""
    cd = _cd(company)
    inv = odoo.search_read("account.move",
        [("state", "=", "posted"), ("move_type", "=", "out_invoice")] + cd,
        ["amount_total", "amount_residual"], limit=0)
    bil = odoo.search_read("account.move",
        [("state", "=", "posted"), ("move_type", "=", "in_invoice")] + cd,
        ["amount_total"], limit=0)

    revenue = sum(i.get("amount_total", 0) or 0 for i in inv)
    ar = sum(i.get("amount_residual", 0) or 0 for i in inv)
    expenses = sum(b.get("amount_total", 0) or 0 for b in bil)

    if revenue == 0 and expenses == 0:
        return "Insufficient accounting data for runway simulation."

    months = max(datetime.now().month, 1)
    burn = max(expenses / months, 100)
    cash = max(revenue - expenses, 0) * 0.15

    sims = sorted([max((cash + ar * random.uniform(0.05, 0.95)) /
                       max(burn / 30 * random.uniform(0.7, 1.5), 0.01), 0)
                   for _ in range(5000)])
    p = lambda q: sims[int(5000 * q)]
    return ("Monte Carlo Runway (N=5000):\n  Cash=${:,.0f} Burn=${:,.0f}/mo AR=${:,.0f}\n"
            "  P5={:.0f}d P25={:.0f}d P50={:.0f}d P75={:.0f}d P95={:.0f}d\n"
            "  Median: {:.1f} months").format(
        cash, burn, ar, p(0.05), p(0.25), p(0.5), p(0.75), p(0.95), p(0.5)/30)


@tool
def compare_periods(
    metric: str = "sales", current_start: str = "", current_end: str = "",
    prior_start: str = "", prior_end: str = "", company: str = "",
) -> str:
    """Compare a metric across two periods (YoY, QoQ). metric: sales|accounting."""
    if not all([current_start, current_end, prior_start, prior_end]):
        return "Provide all 4 dates: current_start, current_end, prior_start, prior_end."
    cd = _cd(company)
    if metric == "sales":
        cur = odoo.search_read("sale.order",
            [("date_order", ">=", current_start), ("date_order", "<=", current_end)] + cd,
            ["amount_total"], limit=0)
        pri = odoo.search_read("sale.order",
            [("date_order", ">=", prior_start), ("date_order", "<=", prior_end)] + cd,
            ["amount_total"], limit=0)
        cr = sum(o.get("amount_total", 0) or 0 for o in cur)
        pr = sum(o.get("amount_total", 0) or 0 for o in pri)
        g = ((cr - pr) / pr * 100) if pr else 0
        return ("Sales: Current({}..{})={} orders ${:,.2f}\n"
                "  Prior({}..{})={} orders ${:,.2f}\n"
                "  Growth: {:+.1f}%  Orders: {:+d}").format(
            current_start, current_end, len(cur), cr,
            prior_start, prior_end, len(pri), pr, g, len(cur)-len(pri))
    elif metric == "accounting":
        cur = odoo.search_read("account.move",
            [("state","=","posted"),("date",">=",current_start),("date","<=",current_end)] + cd,
            ["amount_total", "move_type"], limit=0)
        pri = odoo.search_read("account.move",
            [("state","=","posted"),("date",">=",prior_start),("date","<=",prior_end)] + cd,
            ["amount_total", "move_type"], limit=0)
        ci = sum(m.get("amount_total",0) or 0 for m in cur if _s(m.get("move_type"))=="out_invoice")
        pi = sum(m.get("amount_total",0) or 0 for m in pri if _s(m.get("move_type"))=="out_invoice")
        g = ((ci-pi)/pi*100) if pi else 0
        return "Invoices: Current=${:,.2f} Prior=${:,.2f} Growth={:+.1f}%".format(ci, pi, g)
    return "Use metric='sales' or 'accounting'."


@tool
def web_search_business_context(query: str = "Bangladesh IT outsourcing trends 2025") -> str:
    """Search the web for industry benchmarks, market news, or competitive context."""
    if not tavily:
        return "Web search unavailable: no TAVILY_API_KEY configured."
    try:
        results = tavily.search(query=query, max_results=5)
        lines = ["Web: '{}'".format(query)]
        for r in results.get("results", []):
            content = r.get("content", "")[:300]
            lines.append("\n  {}\n  {}\n  {}".format(r.get("title","N/A"), r.get("url",""), content))
        return "\n".join(lines)
    except Exception as e:
        return "Web search failed: {}".format(e)


# ========================================================================
# 18. SYSTEM PROMPT (Full domain knowledge)
# ========================================================================

SYSTEM_PROMPT = (
    "You are ODIN v3.0 - a senior enterprise intelligence analyst for Betopia Group.\n"
    "You have LIVE read-only access to their Odoo 19 ERP (20+ companies, 63 custom modules, 310+ models).\n\n"
    "YOUR WORKFLOW:\n"
    "1. UNDERSTAND the user's query - what business data are they looking for?\n"
    "2. SEARCH: Call tools to find and retrieve the relevant data.\n"
    "   - If you know the domain, use the specialized tool directly.\n"
    "   - If unsure which model holds the data, call discover_installed_models(keyword) first.\n"
    "   - Then call explore_model_fields(model) to learn field names.\n"
    "   - Then call query_any_model() to retrieve actual records.\n"
    "3. ANALYZE: Compute totals, spot trends, flag anomalies.\n"
    "4. RESPOND: Give a clear, data-backed answer with numbers. Always cite sources.\n\n"
    "TOOLS OVERVIEW:\n\n"
    "MODEL DISCOVERY (use when you don't know which model to query):\n"
    "  - discover_installed_models(keyword) -> list all 120+ custom models, filter by keyword\n"
    "  - explore_model_fields(model_name) -> all fields of any model\n"
    "  - query_any_model(model, domain_json, fields_csv, limit, order, company) -> query ANY model\n"
    "  - count_records(model, domain_json, company) -> fast count\n"
    "  - get_full_data_overview(company) -> record counts for 30+ key models\n"
    "  - list_companies() -> all companies\n\n"
    "SPECIALIZED TOOLS (pre-built aggregations):\n"
    "  - query_sales_orders(status, order_status, date_from, date_to, limit=0, company)\n"
    "    * status: draft|sale|done|cancel|all (Odoo standard state)\n"
    "    * order_status: nra|wip|delivered|complete|cancelled|revisions|issues|all (custom delivery status)\n"
    "  - query_crm_leads(stage, limit=0, company)\n"
    "  - query_operations(status, limit=0, company)\n"
    "  - query_kpi_records(year, quarter, limit=0, company)\n"
    "  - query_accounting(move_type, date_from, date_to, limit=0, company)\n"
    "  - query_profit_and_loss(date_from, date_to, company) -> REAL P&L from journal entries\n"
    "  - query_budget(year, company)\n"
    "  - query_assets(status, company)\n"
    "  - query_payroll(month, year, limit=0, company)\n"
    "  - query_loans(status, company)\n"
    "  - query_attendance(date_from, date_to, limit=0, company)\n"
    "  - query_late_penalties(year, company)\n"
    "  - query_employees(department, limit=0, company)\n"
    "  - query_procurement(status, limit=0, company)\n"
    "  - query_purchase_requisitions(status, company)\n"
    "  - query_helpdesk(status, limit=0, company)\n"
    "  - query_profile_database(limit=0, company)\n"
    "  - query_product_lab(limit=0)\n"
    "  - get_exchange_rates()\n"
    "  - calculate_solvency_score(company)\n"
    "  - run_monte_carlo_runway(company)\n"
    "  - compare_periods(metric, current_start, current_end, prior_start, prior_end, company)\n"
    "  - web_search_business_context(query)\n\n"
    "COMPLETE MODEL CATALOG (310+ models from 63 custom addons):\n\n"
    "  Sales & CRM: sale.order, sale.order.line, crm.lead, bd.platform_source, bd.team,\n"
    "    bd.profile, bd.milestone, bd.order_source, client.reference,\n"
    "    incoming.query.method, lead.status, sales.kpi.dashboard\n\n"
    "  Operations: project.operation, project.task, service.type, assign.team\n"
    "    Note: project.operation uses 'order_status' not 'state'\n\n"
    "  KPI & Bonus: employee.kpi, kpi.grade, kpi.level, kpi.role, kpi.line,\n"
    "    kpi.deduction, kpi.payment, bonus.calculation, bonus.calculation.line,\n"
    "    employee.salaries, nexus.kpi, nexus.kpi.line\n\n"
    "  HR: hr.employee, hr.department, hr.contract, hr.designation, hr.resign,\n"
    "    hr.blood.group, hr.religion, hr.workforce.category, hr.approved.by,\n"
    "    hr.announcement, company.policy, policy.category,\n"
    "    position.change.approver, update.employee.position.designation\n\n"
    "  Attendance: employee.attendance.details, employee.attendance.operations,\n"
    "    employee.attendance.source, employee.monthly.offday, employee.movement,\n"
    "    employee.movement.rule, employee.raw.attendance, employee.weekly.offday,\n"
    "    hr.employee.calendar, hr.employee.roster, hr.employee.roster.line,\n"
    "    hr.late.penalty, roster.attendance, holiday.setup,\n"
    "    shift.change, shift.change.approver\n\n"
    "  Payroll: hr.payslip, hr.payslip.run, hr.payslip.line, hr.payslip.input,\n"
    "    hr.salary.rule, hr.salary.rule.category, hr.payroll.structure,\n"
    "    hr.salary.adjustment, hr.salary.adjustment.history,\n"
    "    hr.salary.adjustment.type, hr.contribution.register,\n"
    "    hr.loan, tax.slab, tax.slab.line, hr.tax.entry, hr.tax.challan,\n"
    "    cost.center, contract.gross.distribution\n\n"
    "  Accounting: account.move, account.move.line, account.payment,\n"
    "    account.journal, account.account, account.tax, account.fiscal.year,\n"
    "    bank.statement.upload, bank.statement.import.row, bank.upload.file,\n"
    "    fee.account.mapping, fiverr.description.account.map, profile.account.mapping,\n"
    "    account.recurring.template, recurring.payment, recurring.payment.line\n\n"
    "  Budget: crossovered.budget, crossovered.budget.lines, account.budget.post\n\n"
    "  Assets: account.asset.asset, account.asset.category, account.asset.depreciation.line\n\n"
    "  Procurement: bp.tender, bp.tender.category, bp.tender.document,\n"
    "    bp.tender.addendum, bp.tender.eval.criteria, bp.tender.line.template,\n"
    "    bp.tender.required.doc, bp.bid, bp.bid.line, bp.bid.file,\n"
    "    bp.bill, bp.challan, bp.award.decision, bp.vendor.company,\n"
    "    bp.vendor.document, bp.vendor.owner, bp.evaluation.score,\n"
    "    bp.evaluation.session, bp.ledger.entry, bp.audit.log,\n"
    "    bp.notification, bp.clarification.thread, bp.clarification.message\n\n"
    "  Purchase: cus.purchase.requisition, requisition.order.line\n\n"
    "  Helpdesk: ticket.helpdesk, ticket.stage, support.ticket,\n"
    "    helpdesk.category, helpdesk.tag, helpdesk.type, team.helpdesk\n\n"
    "  Profile DB: profile.db, fiverr.information, upwork.information,\n"
    "    kwork.information, freelancer.information, payoneer.information,\n"
    "    pph.information, stripe.information, wise.information,\n"
    "    remitly.information, nsave.information, nsave.payoneer.information,\n"
    "    priyo.pay.information, proyo.payoneer.information,\n"
    "    employee.sim.info, profile.ip.info, deleted.profile.log\n\n"
    "  Marketplace: profile.profile, profile.transaction,\n"
    "    marketplace.marketplace, marketplace.particular\n\n"
    "  Product Lab: product.lab.project, product.lab.industry,\n"
    "    product.lab.technology, product.lab.document.type,\n"
    "    product.lab.project.document, product.lab.project.feature,\n"
    "    product.lab.project.image, product.lab.project.technology\n\n"
    "  Announcements: portal.announcement, announcement.reaction,\n"
    "    announcement.read.status\n\n"
    "  Dashboard: analytics.dashboard, operations.analytics, sales.analytics\n\n"
    "  Chatbot: chatbot.config, chatbot.message, chatbot.tag\n\n"
    "  Proposal: proposal.template\n\n"
    "  Standard: res.company, res.partner, res.users, res.currency,\n"
    "    product.template, product.product\n\n"
    "DOMAIN KNOWLEDGE:\n"
    "  - Betopia Group: Bangladesh-based IT services company with 20+ subsidiary companies\n"
    "  - Multi-company: Every query can be filtered by company name or ID\n"
    "  - Currencies: BDT (Taka) and USD. KPI bonuses in BDT, sales in USD.\n"
    "  - Freelancer platforms: Fiverr, Upwork, Kwork, Freelancer.com, PPH\n"
    "  - Current date: " + datetime.now().strftime("%Y-%m-%d") + "\n\n"
    "IMPORTANT RULES:\n"
    "  - For P&L / Net Profit / Income Statement, ALWAYS use query_profit_and_loss (NOT query_accounting)\n"
    "  - query_accounting shows invoice/bill summaries; query_profit_and_loss shows real P&L from journal entries\n"
    "  - ALL aggregation tools default to limit=0 (fetch ALL records)\n"
    "  - ALWAYS cite specific numbers from tool results\n"
    "  - When unsure of field names, call explore_model_fields FIRST\n"
    "  - When unsure which model to use, call discover_installed_models FIRST\n"
    "  - For multi-company analysis, call list_companies first\n"
    "  - Flag anomalies and explain business impact\n"
    "  - End with prioritized action items\n"
)


# ========================================================================
# 19. AGENT GRAPH
# ========================================================================

ALL_TOOLS = [
    # Exploration & Discovery
    discover_installed_models, get_full_data_overview, list_companies,
    explore_model_fields, query_any_model, count_records,
    # Sales & CRM
    query_sales_orders, query_crm_leads,
    # Operations
    query_operations,
    # KPI
    query_kpi_records,
    # Accounting
    query_accounting, query_profit_and_loss, query_budget, query_assets,
    # Payroll
    query_payroll, query_loans,
    # Attendance
    query_attendance, query_late_penalties,
    # HR
    query_employees,
    # Procurement
    query_procurement, query_purchase_requisitions,
    # Helpdesk
    query_helpdesk,
    # Profiles
    query_profile_database,
    # Product Lab
    query_product_lab,
    # Analytical
    get_exchange_rates, calculate_solvency_score, run_monte_carlo_runway,
    compare_periods,
    # External
    web_search_business_context,
]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_agent():
    llm = ChatOpenAI(
        model="gpt-4o", temperature=0.1,
        api_key=OPENAI_API_KEY,
    ).bind_tools(ALL_TOOLS)

    tool_node = ToolNode(ALL_TOOLS)

    def agent_node(state):
        msgs = list(state["messages"])
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {"messages": [llm.invoke(msgs)]}

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


# ========================================================================
# 20. INTERACTIVE SESSION
# ========================================================================

def run_interactive():
    print("=" * 70)
    print("  ODIN AGENTIC ADVISOR v3.0")
    print("  Betopia Group - Full Enterprise Intelligence")
    tool_count = len(ALL_TOOLS)
    print("  {} Tools | 20+ Companies | 310+ Models | ReAct Agent".format(tool_count))
    print("=" * 70)
    print()
    print("  Search your entire business - just type what you need:")
    print()
    print("  Examples:")
    print("    - Give me a full business health check")
    print("    - Show all companies and their employee counts")
    print("    - What is the payroll cost breakdown this year?")
    print("    - How many helpdesk tickets are open?")
    print("    - Show procurement tender pipeline")
    print("    - Compare Q1 2025 vs Q1 2024 sales")
    print("    - Which employees have the most late penalties?")
    print("    - Show me freelancer profile distribution across platforms")
    print("    - Who are the top 10 salesmen by revenue?")
    print("    - Show all Fiverr profiles and their status")
    print("    - What models are available for attendance?")
    print()
    print("  Type 'quit' to exit.")
    print("-" * 70)

    agent = create_agent()
    thread = {"configurable": {"thread_id": "odin-v3-session"}}
    turn = 0

    while True:
        print()
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("\nODIN closed.")
            break

        turn += 1
        print("\n" + "-" * 70)
        print("ODIN analyzing (turn {})...\n".format(turn))
        try:
            result = agent.invoke({"messages": [HumanMessage(content=q)]}, thread)
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    print("ODIN:\n{}".format(msg.content))
                    break
        except Exception as e:
            logger.error("Agent error: {}".format(e), exc_info=True)
            print("\n  Error: {}".format(e))


if __name__ == "__main__":
    run_interactive()
