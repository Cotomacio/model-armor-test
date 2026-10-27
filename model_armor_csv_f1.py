#!/usr/bin/env python3
"""
Model Armor CSV Tester with Intelligent Category Grouping (F1 Score)
------------------------------------------------------------------
This script tests prompts and calculates F1 Scores by grouping
similar threats (e.g., Harassment, Hate Speech) into their official
Model Armor filter families (e.g., RESPONSIBLE_AI).

It computes both an overall (binary) F1 and a per-category F1 when
the CSV provides a Category column for attack rows.

Usage:
  python model_armor_csv_f1.py --project_id YOUR_PROJECT --template_id YOUR_TEMPLATE --location us-central1 --csv_file data.csv
"""

import csv
import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, Any, Set, List, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
import google.auth
from google.auth.transport.requests import Request

# Official Model Armor filter families this script reports on.
KNOWN_CATEGORIES = ('PROMPT_INJECTION', 'RESPONSIBLE_AI', 'PII', 'MALICIOUS_URIS', 'CSAM')

# Accepted aliases for the CSV Category column. Each maps to one of KNOWN_CATEGORIES.
# Comparison is case-insensitive (input is upper()-cased before lookup).
CATEGORY_ALIASES: Dict[str, Tuple[str, ...]] = {
    'PROMPT_INJECTION': ('PROMPT_INJECTION', 'JAILBREAK', 'PI_AND_JAILBREAK'),
    'RESPONSIBLE_AI':   ('RESPONSIBLE_AI', 'RAI', 'TOXICITY', 'HATE_SPEECH',
                         'HARASSMENT', 'DANGEROUS', 'SEXUALLY_EXPLICIT'),
    'PII':              ('PII', 'SDP', 'SENSITIVE_DATA'),
    'MALICIOUS_URIS':   ('MALICIOUS_URIS', 'MALICIOUS_URI', 'URIS'),
    'CSAM':             ('CSAM',),
}
VALID_CATEGORY_ALIASES = frozenset(a for aliases in CATEGORY_ALIASES.values() for a in aliases)

# Per-filter token limits (https://cloud.google.com/model-armor/quotas). Prompts
# above a filter's limit make that filter return EXECUTION_SKIPPED instead of a
# verdict. Tokens are estimated as chars/4 — a warning heuristic, not an exact count.
EST_CHARS_PER_TOKEN = 4
FILTER_TOKEN_LIMITS = {
    'PROMPT_INJECTION': 10_000,
    'RESPONSIBLE_AI': 10_000,
    'CSAM': 10_000,
    'PII': 130_000,
}


class ModelArmorClient:
    def __init__(self, project_id: str, template_id: str, location: str,
                 verbose: bool = False, max_workers: int = 1):
        self.project_id = project_id
        self.template_id = template_id
        self.location = location
        self.verbose = verbose
        self._credentials = None
        self._auth_lock = threading.Lock()
        self._latency_lock = threading.Lock()
        self._session = requests.Session()
        pool_size = max(4, max_workers)
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        # Latency of each successful HTTP call (in milliseconds).
        self.latencies_ms: List[float] = []

    def _get_credentials(self):
        if self._credentials is None:
            try:
                self._credentials, _ = google.auth.default()
            except Exception as e:
                # Fallback: try gcloud from PATH (no hardcoded path).
                gcloud = shutil.which("gcloud")
                if not gcloud:
                    raise RuntimeError(
                        "Failed to obtain ADC credentials and 'gcloud' was not found on PATH."
                    ) from e
                try:
                    res = subprocess.run(
                        [gcloud, "auth", "application-default", "print-access-token"],
                        capture_output=True, text=True, check=True, timeout=15,
                    )
                    token = res.stdout.strip()
                    if not token:
                        raise RuntimeError("gcloud returned an empty token.")
                    # Minimal wrapper that behaves like Credentials for the rest of the flow.
                    self._credentials = _StaticTokenCredentials(token)
                except subprocess.SubprocessError as err:
                    raise RuntimeError(f"Failed to execute gcloud: {err}") from e
        return self._credentials

    def _access_token(self) -> str:
        with self._auth_lock:
            creds = self._get_credentials()
            # Proactive refresh if the token is expired or about to expire.
            if getattr(creds, "expired", False) or not getattr(creds, "token", None):
                try:
                    creds.refresh(Request())
                except Exception:
                    # For static credentials (gcloud fallback) refresh is a no-op.
                    pass
            return creds.token

    def _build_headers(self, json_content: bool = True) -> Dict[str, str]:
        token = self._access_token()
        headers: Dict[str, str] = {"Authorization": f"Bearer {token}"}
        if json_content:
            headers["Content-Type"] = "application/json"
        creds = self._credentials
        quota_project = getattr(creds, "quota_project_id", None) if creds is not None else None
        if quota_project:
            headers["x-goog-user-project"] = quota_project
        return headers

    def sanitize(self, prompt: str) -> Dict[str, Any]:
        url = (
            f"https://modelarmor.{self.location}.rep.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"templates/{self.template_id}:sanitizeUserPrompt"
        )
        payload = {"user_prompt_data": {"text": prompt}}

        backoff = 1.0
        last_response = None
        last_exception_type: str = ""
        last_exception_msg: str = ""

        for attempt in range(4):
            try:
                headers = self._build_headers(json_content=True)
                start = time.perf_counter()
                response = self._session.post(url, headers=headers, json=payload, timeout=30)
                latency_ms = (time.perf_counter() - start) * 1000.0
                last_response = response

                if self.verbose:
                    print(f"   [verbose] attempt {attempt + 1}/4 → HTTP {response.status_code} ({latency_ms:.0f} ms)")

                if response.status_code == 401:
                    # Token rejected: force a refresh on the next iteration.
                    with self._auth_lock:
                        self._credentials = None
                    continue
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue

                response.raise_for_status()
                with self._latency_lock:
                    self.latencies_ms.append(latency_ms)
                data = response.json()
                if isinstance(data, dict):
                    data["_latency_ms"] = latency_ms
                return data
            except requests.RequestException as e:
                last_exception_type = type(e).__name__
                last_exception_msg = str(e)
                if self.verbose:
                    print(f"   [verbose] attempt {attempt + 1}/4 → {last_exception_type}: {last_exception_msg}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue

        # All attempts failed. Return rich diagnostic info for the caller.
        if last_response is not None:
            body = (last_response.text or "")[:500]
            return {
                "is_error": True,
                "error_status": last_response.status_code,
                "error_body": body,
                "error_url": url,
            }
        return {
            "is_error": True,
            "error_status": "no_response",
            "error_type": last_exception_type or "Unknown",
            "error_message": last_exception_msg or "(no exception captured)",
            "error_url": url,
        }

    def describe_template(self) -> Dict[str, Any]:
        """Fetch the template definition so the report records the exact
        configuration the numbers were measured against."""
        url = (
            f"https://modelarmor.{self.location}.rep.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"templates/{self.template_id}"
        )
        headers = self._build_headers(json_content=False)
        response = self._session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()


class _StaticTokenCredentials:
    """Minimal wrapper for a static token obtained via the `gcloud` subprocess."""

    def __init__(self, token: str):
        self.token = token
        self.expired = False

    def refresh(self, request):  # noqa: ARG002
        # No-op: static tokens cannot be refreshed here.
        return None


def map_category(raw_category: str) -> str:
    """Map a label to the official Model Armor filter family. Returns the
    raw input unchanged if no alias matches (callers should validate first
    via VALID_CATEGORY_ALIASES if strictness is required)."""
    cat = raw_category.upper().strip()
    for family, aliases in CATEGORY_ALIASES.items():
        if cat in aliases:
            return family
    return cat


def extract_filter_states(response_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-category filter outcome:
    {'match': bool, 'skipped': bool, 'absent': bool, 'confidence': str|None}.

    'skipped' means the filter returned EXECUTION_SKIPPED (e.g. the prompt
    exceeded its token limit): the filter did NOT evaluate the prompt, which is
    different from evaluating it and finding no match.
    'absent' means the filter did not appear in the API response at all —
    typically because it is not configured in the template. Treating that as a
    True Negative would disguise "never ran" as "ran and correctly allowed"."""
    filter_results = response_data.get('sanitizationResult', {}).get('filterResults', {})

    def state(d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match': d.get('matchState') == 'MATCH_FOUND',
            'skipped': d.get('executionState') == 'EXECUTION_SKIPPED',
            'absent': not d,
            'confidence': d.get('confidenceLevel'),
        }

    states: Dict[str, Dict[str, Any]] = {}

    pi_jb = filter_results.get('pi_and_jailbreak', {}).get('piAndJailbreakFilterResult', {})
    states['PROMPT_INJECTION'] = state(pi_jb)

    rai = filter_results.get('rai', {}).get('raiFilterResult', {})
    rai_state = state(rai)
    matched_types = [
        f"{rai_type}:{rai_data.get('confidenceLevel', '?')}"
        for rai_type, rai_data in rai.get('raiFilterTypeResults', {}).items()
        if rai_data.get('matchState') == 'MATCH_FOUND'
    ]
    rai_state['match'] = bool(matched_types)
    rai_state['confidence'] = ','.join(matched_types) if matched_types else None
    states['RESPONSIBLE_AI'] = rai_state

    # SDP: basic templates answer in inspectResult; advanced templates with a
    # deidentify template answer in deidentifyResult instead.
    sdp = filter_results.get('sdp', {}).get('sdpFilterResult', {})
    inspect = sdp.get('inspectResult', {})
    deidentify = sdp.get('deidentifyResult', {})
    states['PII'] = {
        'match': 'MATCH_FOUND' in (inspect.get('matchState'), deidentify.get('matchState')),
        'skipped': 'EXECUTION_SKIPPED' in (inspect.get('executionState'), deidentify.get('executionState')),
        'absent': not sdp,
        'confidence': None,
    }

    csam = filter_results.get('csam', {}).get('csamFilterFilterResult', {})
    states['CSAM'] = state(csam)

    uris = filter_results.get('malicious_uris', {}).get('maliciousUriFilterResult', {})
    states['MALICIOUS_URIS'] = state(uris)

    return states


def print_template_config(client: ModelArmorClient):
    """Print the template's filter configuration so every report is tied to the
    exact settings it was measured against."""
    try:
        tpl = client.describe_template()
    except Exception as e:
        print(f"\n⚠️  Could not fetch the template configuration ({e}) — continuing without it.")
        return

    fc = tpl.get('filterConfig', {})
    meta = tpl.get('templateMetadata', {})
    print("\n🧩 TEMPLATE CONFIGURATION")
    print(f"   Template : {tpl.get('name', client.template_id)}")

    pi = fc.get('piAndJailbreakFilterSettings', {})
    if pi.get('filterEnforcement') == 'ENABLED':
        print(f"   Prompt injection & jailbreak : ENABLED ({pi.get('confidenceLevel', 'default')})")
    else:
        print("   Prompt injection & jailbreak : disabled")

    sdp = fc.get('sdpSettings', {})
    if sdp.get('basicConfig', {}).get('filterEnforcement') == 'ENABLED':
        print("   Sensitive Data Protection    : BASIC mode")
        print("     ⚠️  Basic mode only detects credit cards, financial account numbers,")
        print("        GCP credentials/API keys and passwords (plus US SSN/ITIN in US")
        print("        regions). Emails, phone numbers and country-specific IDs (e.g.")
        print("        BRAZIL_CPF_NUMBER) require ADVANCED mode with a DLP inspect template.")
    elif 'advancedConfig' in sdp:
        adv = sdp['advancedConfig']
        print("   Sensitive Data Protection    : ADVANCED mode")
        if adv.get('inspectTemplate'):
            print(f"     inspect template   : {adv['inspectTemplate']}")
        if adv.get('deidentifyTemplate'):
            print(f"     deidentify template: {adv['deidentifyTemplate']}")
    else:
        print("   Sensitive Data Protection    : disabled")

    rai = fc.get('raiSettings', {}).get('raiFilters', [])
    if rai:
        rai_str = ', '.join(f"{r.get('filterType')}:{r.get('confidenceLevel', 'default')}" for r in rai)
        print(f"   Responsible AI               : {rai_str}")
    else:
        print("   Responsible AI               : disabled")

    uri = fc.get('maliciousUriFilterSettings')
    if uri is not None:
        print(f"   Malicious URIs               : {uri.get('filterEnforcement', '?')}")
    csam = fc.get('csamFilterSettings')
    if csam is not None:
        print(f"   CSAM                         : {csam.get('filterEnforcement', '?')}")

    extras = []
    if meta.get('enforcementType'):
        extras.append(f"enforcement={meta['enforcementType']}")
    alias = meta.get('filterVersionSelector', {}).get('alias')
    if alias:
        extras.append(f"filter_version={alias}")
    mld = meta.get('multiLanguageDetection', {}).get('enableMultiLanguageDetection')
    if mld is not None:
        extras.append(f"multi_language={'on' if mld else 'off'}")
    if extras:
        print(f"   Metadata                     : {', '.join(extras)}")


def percentile(values: List[float], p: float) -> float:
    """Linear-interpolation percentile (same method as numpy default)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float, float, int]:
    """Compute Precision, Recall and F1 Score."""
    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, total


def parse_expected_categories(raw: str) -> Set[str]:
    """Parse the Category column from the CSV (supports multiple categories separated by '|' or ',')."""
    if not raw:
        return set()
    parts = [p for chunk in raw.split('|') for p in chunk.split(',')]
    return {map_category(p) for p in parts if p.strip()}


def _format_accepted_categories() -> str:
    lines = ["Accepted Category values (case-insensitive):"]
    for family, aliases in CATEGORY_ALIASES.items():
        lines.append(f"   {family:<20} → {', '.join(aliases)}")
    lines.append("Leave the Category column empty for safe rows.")
    return "\n".join(lines)


def run_csv_test(client: ModelArmorClient, filepath: str, request_delay: float = 0.0,
                 category_report: bool = True, concurrency: int = 1,
                 output_csv: Optional[str] = None):
    prompts_data: List[Dict[str, Any]] = []
    invalid_labels: List[Tuple[int, str]] = []
    safe_with_category: List[Tuple[int, str]] = []
    attack_missing_category: List[int] = []

    skipped_empty = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                print("❌ CSV is empty (no header row).")
                sys.exit(1)
            # Resolve columns by header name so column order doesn't matter.
            names = [h.strip().lower() for h in header]
            prompt_idx = next((i for i, n in enumerate(names) if n.startswith('prompt')), None)
            attack_idx = next((i for i, n in enumerate(names) if n.startswith('is_attack')), None)
            if prompt_idx is None or attack_idx is None:
                print(f"❌ CSV header must contain 'Prompt' and 'Is_Attack' columns (found: {', '.join(header)}).")
                sys.exit(1)
            category_idx = next((i for i, n in enumerate(names) if n.startswith('categor')), None)

            # enumerate from 2 because the header is row 1.
            for row_num, row in enumerate(reader, start=2):
                if len(row) <= max(prompt_idx, attack_idx):
                    continue
                prompt_text = row[prompt_idx].strip()
                if not prompt_text:
                    # A row like `,,` would otherwise be sent to the API as a
                    # billable "safe" prompt and counted as a bogus TN.
                    skipped_empty += 1
                    continue
                is_attack_str = row[attack_idx].strip().lower()
                is_attack = is_attack_str in ('yes', 'y', 'true', '1')
                expected_cats: Set[str] = set()
                raw_cat = ""
                if category_idx is not None and len(row) > category_idx:
                    raw_cat = row[category_idx].strip()
                    if raw_cat:
                        tokens = [p.strip() for chunk in raw_cat.split('|') for p in chunk.split(',') if p.strip()]
                        for tok in tokens:
                            if tok.upper() not in VALID_CATEGORY_ALIASES:
                                invalid_labels.append((row_num, tok))
                        expected_cats = parse_expected_categories(raw_cat)

                if not is_attack and raw_cat:
                    safe_with_category.append((row_num, raw_cat))
                elif is_attack and not raw_cat:
                    attack_missing_category.append(row_num)

                prompts_data.append({
                    "row_num": row_num,
                    "text": prompt_text,
                    "is_attack": is_attack,
                    "expected_categories": expected_cats,
                })
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    if invalid_labels:
        print(f"❌ Invalid Category labels in {filepath} — aborting before evaluation.\n")
        for row_num, tok in invalid_labels:
            print(f"   row {row_num}: {tok!r}")
        print()
        print(_format_accepted_categories())
        sys.exit(1)

    if safe_with_category:
        print(f"❌ Safe rows (Is_Attack=no) must not have a Category in {filepath} — aborting before evaluation.\n")
        for row_num, raw_cat in safe_with_category:
            print(f"   row {row_num}: Category={raw_cat!r} on a safe row")
        print("\nLeave the Category column empty for safe rows (otherwise per-category metrics would treat them as expected attacks).")
        sys.exit(1)

    has_any_expected = any(p['expected_categories'] for p in prompts_data)
    if has_any_expected and category_report and attack_missing_category:
        print(f"❌ Attack rows (Is_Attack=yes) with an empty Category in {filepath} — aborting before evaluation.\n")
        for row_num in attack_missing_category:
            print(f"   row {row_num}: Is_Attack=yes but Category is empty")
        print(
            "\nWhen using per-category evaluation, every attack row must specify its expected Category\n"
            "(otherwise a valid block on that row is counted as a per-category False Positive).\n"
            "Fill in the Category column for the row(s) above, or pass --no_category_report for binary evaluation only."
        )
        sys.exit(1)

    if skipped_empty:
        print(f"\nℹ️  {skipped_empty} row(s) with an empty Prompt were skipped.")

    # Warn upfront about prompts that likely exceed per-filter token limits.
    oversized: List[Tuple[int, int, List[str]]] = []
    for idx, p in enumerate(prompts_data, start=1):
        est_tokens = len(p['text']) // EST_CHARS_PER_TOKEN
        at_risk = [cat for cat, limit in FILTER_TOKEN_LIMITS.items() if est_tokens > limit]
        if at_risk:
            oversized.append((idx, est_tokens, at_risk))
    if oversized:
        print(f"\n⚠️  {len(oversized)} prompt(s) may exceed per-filter token limits")
        print(f"   (~{EST_CHARS_PER_TOKEN} chars/token estimate). Affected filters return EXECUTION_SKIPPED")
        print("   instead of a verdict — reported separately (SKIP column), not as PASSED:")
        for idx, est_tokens, at_risk in oversized:
            print(f"   prompt #{idx}: ~{est_tokens} tokens → {', '.join(at_risk)}")

    print_template_config(client)

    print(f"\n🚀 Starting test ({len(prompts_data)} prompts)")
    print("-" * 110)
    print(f"{'PROMPT (Preview)':<45} | {'EXPECTED':<17} | {'DETECTED':<10} | {'EVALUATION'}")
    print("-" * 110)

    g_tp = g_fp = g_fn = g_tn = 0
    g_errors = 0
    g_prompts_with_skips = 0
    failed_prompts: List[Dict[str, Any]] = []
    export_rows: List[Dict[str, Any]] = []
    # Per-category confusion matrix. 'skip' counts prompts the filter did not
    # evaluate (EXECUTION_SKIPPED); 'na' counts responses where the filter was
    # absent (not configured in the template). Both are excluded from the
    # matrix. 'exp' counts prompts whose ground truth expected the category.
    cat_stats: Dict[str, Dict[str, int]] = {
        c: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'skip': 0, 'na': 0, 'exp': 0}
        for c in KNOWN_CATEGORIES
    }

    workers = max(1, concurrency)
    if workers > 1 and prompts_data:
        # Warm up credentials once before spawning worker threads.
        try:
            client._access_token()
        except Exception:
            pass
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = []
        for idx, p_data in enumerate(prompts_data):
            futures.append(executor.submit(client.sanitize, p_data['text']))
            if request_delay and idx + 1 < len(prompts_data):
                time.sleep(request_delay)
    else:
        executor = None
        futures = []

    try:
        for idx, p_data in enumerate(prompts_data, start=1):
            prompt = p_data['text']
            is_attack_expected = p_data['is_attack']
            expected_cats: Set[str] = p_data['expected_categories']

            if executor is not None:
                response_data = futures[idx - 1].result()
            else:
                response_data = client.sanitize(prompt)

            if response_data.get('is_error'):
                g_errors += 1
                err_status = response_data.get('error_status')
                if err_status == 'no_response':
                    err_short = f"no_response: {response_data.get('error_type', '?')}"
                else:
                    err_short = f"HTTP {err_status}"
                eval_str = f"⚠️  ERROR ({err_short})"
                exp_str = "ATTACK" if is_attack_expected else "SAFE"
                det_str = "—"
                failed_prompts.append({
                    'prompt': prompt,
                    'status': err_status,
                    'type': response_data.get('error_type'),
                    'message': response_data.get('error_message'),
                    'body': response_data.get('error_body'),
                    'url': response_data.get('error_url'),
                })
                export_rows.append({
                    "Row": p_data.get("row_num", idx),
                    "Prompt": prompt,
                    "Expected_Attack": "yes" if is_attack_expected else "no",
                    "Expected_Categories": "|".join(sorted(expected_cats)),
                    "Detected": "ERROR",
                    "Evaluation": "ERROR",
                    "Matched_Filters": "",
                    "Skipped_Filters": "",
                    "Latency_ms": "",
                    "Error_Status": str(err_status),
                })
                prompt_preview = (prompt.replace('\n', ' ')[:42] + '...') if len(prompt) > 42 else prompt.replace('\n', ' ')
                print(f"{prompt_preview:<45} | {exp_str:<17} | {det_str:<10} | {eval_str}")
                if executor is None and request_delay:
                    time.sleep(request_delay)
                continue

            is_blocked = response_data.get('sanitizationResult', {}).get('filterMatchState') == 'MATCH_FOUND'
            filter_states = extract_filter_states(response_data)

            # Overall binary metrics.
            if is_attack_expected and is_blocked:
                eval_str = "✅ TP (Correct Block)"
                eval_code = "TP"
                g_tp += 1
            elif not is_attack_expected and is_blocked:
                eval_str = "❌ FP (False Alarm)"
                eval_code = "FP"
                g_fp += 1
            elif is_attack_expected and not is_blocked:
                eval_str = "❌ FN (Attack Passed!)"
                eval_code = "FN"
                g_fn += 1
            else:
                eval_str = "✅ TN (Correctly Allowed)"
                eval_code = "TN"
                g_tn += 1

            # Per-category metrics (each category is an independent binary classifier).
            for cat in KNOWN_CATEGORIES:
                st = filter_states[cat]
                if cat in expected_cats:
                    cat_stats[cat]['exp'] += 1
                if st['absent']:
                    # Filter missing from the response (not configured in the
                    # template) — counting it as TN would disguise "never ran"
                    # as "ran and correctly allowed".
                    cat_stats[cat]['na'] += 1
                    continue
                if st['skipped']:
                    # The filter never evaluated this prompt — neither a hit nor a miss.
                    cat_stats[cat]['skip'] += 1
                    continue
                expected = cat in expected_cats
                predicted = st['match']
                if expected and predicted:
                    cat_stats[cat]['tp'] += 1
                elif not expected and predicted:
                    cat_stats[cat]['fp'] += 1
                elif expected and not predicted:
                    cat_stats[cat]['fn'] += 1
                else:
                    cat_stats[cat]['tn'] += 1

            matched_detail = [
                f"{cat}:{filter_states[cat]['confidence']}" if filter_states[cat]['confidence'] else cat
                for cat in KNOWN_CATEGORIES if filter_states[cat]['match']
            ]
            skipped_filters = [cat for cat in KNOWN_CATEGORIES if filter_states[cat]['skipped']]
            if skipped_filters:
                g_prompts_with_skips += 1
            detail_str = ""
            if matched_detail:
                detail_str += f"  [{', '.join(matched_detail)}]"
            if skipped_filters:
                detail_str += f"  [SKIPPED: {', '.join(skipped_filters)}]"

            prompt_preview = (prompt.replace('\n', ' ')[:42] + '...') if len(prompt) > 42 else prompt.replace('\n', ' ')
            exp_str = "ATTACK" if is_attack_expected else "SAFE"
            det_str = "BLOCKED" if is_blocked else "PASSED"
            print(f"{prompt_preview:<45} | {exp_str:<17} | {det_str:<10} | {eval_str}{detail_str}")

            latency_val = response_data.get("_latency_ms")
            export_rows.append({
                "Row": p_data.get("row_num", idx),
                "Prompt": prompt,
                "Expected_Attack": "yes" if is_attack_expected else "no",
                "Expected_Categories": "|".join(sorted(expected_cats)),
                "Detected": det_str,
                "Evaluation": eval_code,
                "Matched_Filters": "|".join(matched_detail),
                "Skipped_Filters": "|".join(skipped_filters),
                "Latency_ms": f"{latency_val:.1f}" if latency_val is not None else "",
                "Error_Status": "",
            })

            if executor is None and request_delay:
                time.sleep(request_delay)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print("\n" + "=" * 80)
    print("📊 MODEL ARMOR EVALUATION REPORT")
    print("=" * 80)

    print("\n📖 METRICS LEGEND (Confusion Matrix):")
    print("   [TP] True Positive  : Actual attack correctly BLOCKED (Success 🛡️)")
    print("   [TN] True Negative  : Safe prompt correctly ALLOWED (Success ✅)")
    print("   [FP] False Positive : Safe prompt incorrectly BLOCKED (False Alarm / Friction ⚠️)")
    print("   [FN] False Negative : Actual attack incorrectly ALLOWED (Security Failure 🚨)")
    print("-" * 80)

    g_prec, g_rec, g_f1, g_tot = calculate_metrics(g_tp, g_fp, g_fn, g_tn)
    print(f"\n🛡️  OVERALL PERFORMANCE (Any Block)")
    print(f"   Total Prompts Evaluated  : {g_tot}")
    print(f"   API Errors (excluded)    : {g_errors}")
    if g_prompts_with_skips:
        print(f"   Prompts w/ skipped filter: {g_prompts_with_skips} (some filters returned EXECUTION_SKIPPED — see SKIP column)")
    print(f"   Base Metrics    : TP={g_tp} | TN={g_tn} | FP={g_fp} | FN={g_fn}")
    print(f"   Precision       : {g_prec:.4f}")
    print(f"   Recall          : {g_rec:.4f}")
    print(f"   Overall F1 Score: {g_f1:.4f}")

    # Per-category F1 (only if the CSV provides category labels).
    if not category_report:
        print("\n(ℹ️  Per-category report suppressed via --no_category_report.)")
    elif has_any_expected:
        print("\n" + "-" * 80)
        print("📂 PERFORMANCE BY CATEGORY")
        print("-" * 80)
        print(f"{'CATEGORY':<20} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | {'SKIP':>4} | {'PREC':>6} | {'REC':>6} | {'F1':>6}")
        f1_values = []
        any_not_configured = False
        for cat in KNOWN_CATEGORIES:
            s = cat_stats[cat]
            evaluated = s['tp'] + s['fp'] + s['fn'] + s['tn'] + s['skip']
            if evaluated == 0 and s['na'] > 0:
                # The filter never appeared in any response: not configured in
                # the template. Print dashes instead of a bogus TN row.
                any_not_configured = True
                d = '—'
                line = (f"{cat:<20} | {d:>4} | {d:>4} | {d:>4} | {d:>4} | {d:>4} | "
                        f"{d:>6} | {d:>6} | {d:>6}   (not configured in template)")
                if s['exp'] > 0:
                    line += f" ⚠️  {s['exp']} prompt(s) expected this category!"
                print(line)
                continue
            prec, rec, f1, _ = calculate_metrics(s['tp'], s['fp'], s['fn'], s['tn'])
            # Only include categories with at least one expected or predicted positive in the macro-F1.
            if s['tp'] + s['fp'] + s['fn'] > 0:
                f1_values.append(f1)
            line = (f"{cat:<20} | {s['tp']:>4} | {s['fp']:>4} | {s['fn']:>4} | {s['tn']:>4} | "
                    f"{s['skip']:>4} | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f}")
            if s['na']:
                line += f"   (absent in {s['na']} response(s))"
            print(line)

        if f1_values:
            macro_f1 = sum(f1_values) / len(f1_values)
            print(f"\n   Macro F1 (mean across categories with signal): {macro_f1:.4f}")

        if any(s['skip'] for s in cat_stats.values()):
            print("\n   ⚠️  SKIP = the filter returned EXECUTION_SKIPPED (did not evaluate the")
            print("      prompt, e.g. token limit exceeded). Skipped prompts are excluded from")
            print("      that category's confusion matrix — they are neither hits nor misses.")
        if any_not_configured:
            print("\n   ℹ️  '—' = the filter never appeared in any API response, i.e. it is not")
            print("      configured in this template. These rows carry no signal — they are")
            print("      not True Negatives.")
    else:
        print("\n(ℹ️  Category column missing or empty in CSV — per-category F1 not computed.)")

    # Latency report (successful HTTP calls only).
    print("\n" + "-" * 80)
    print("⏱️  MODEL ARMOR API CALL LATENCY")
    print("-" * 80)
    samples = client.latencies_ms
    if not samples:
        print("(No latency samples — no successful calls.)")
    else:
        n = len(samples)
        mean_ms = sum(samples) / n
        print(f"   Samples (HTTP 2xx): {n}")
        print(f"   min  : {min(samples):>8.1f} ms")
        print(f"   mean : {mean_ms:>8.1f} ms")
        print(f"   p50  : {percentile(samples, 50):>8.1f} ms")
        print(f"   p90  : {percentile(samples, 90):>8.1f} ms")
        print(f"   p95  : {percentile(samples, 95):>8.1f} ms")
        print(f"   p99  : {percentile(samples, 99):>8.1f} ms")
        print(f"   max  : {max(samples):>8.1f} ms")

    # Per-failure diagnostics (only printed when there were errors).
    if failed_prompts:
        print("\n" + "=" * 80)
        print(f"🔍 ERROR DIAGNOSTICS  ({len(failed_prompts)} failed prompt(s))")
        print("=" * 80)
        for i, f in enumerate(failed_prompts, start=1):
            preview = f['prompt'].replace('\n', ' ')
            preview = (preview[:80] + '...') if len(preview) > 80 else preview
            print(f"\n#{i}  Prompt : {preview!r}")

            if f['status'] == 'no_response':
                print(f"    Result : No HTTP response after 4 attempts")
                print(f"    Type   : {f.get('type') or '(unknown)'}")
                if f.get('message'):
                    msg = f['message']
                    if len(msg) > 400:
                        msg = msg[:400] + '...'
                    print(f"    Detail : {msg}")
                print(f"    URL    : {f.get('url', '(unknown)')}")
                print(f"\n    Likely causes:")
                print(f"      • Invalid --location value. Multi-region: us, eu.")
                print(f"        Single-region: us-central1, us-east1, us-east4, us-west1,")
                print(f"        europe-west1/2/3/4/9, europe-southwest1, asia-northeast1/3,")
                print(f"        asia-south1, asia-southeast1, australia-southeast2,")
                print(f"        northamerica-northeast2.")
                print(f"      • Network / firewall blocking outbound HTTPS to *.googleapis.com.")
                print(f"      • DNS resolution failure (check the URL above is reachable).")
                print(f"      • If running inside a VPC, configure Private Service Connect.")
                continue

            print(f"    Result : HTTP {f['status']}")
            if f.get('body'):
                print(f"    Body   : {f['body']}")
            print(f"    URL    : {f.get('url', '(unknown)')}")

            print(f"\n    Likely causes:")
            if f['status'] == 401:
                print(f"      • ADC token rejected. Run: gcloud auth application-default login")
            elif f['status'] == 403:
                print(f"      • Account lacks roles/modelarmor.user on this project.")
                print(f"      • ADC quota project mismatch. Run:")
                print(f"        gcloud auth application-default set-quota-project YOUR_PROJECT_ID")
            elif f['status'] == 404:
                print(f"      • --location or --template_id is wrong, or the template was")
                print(f"        deleted. Verify at:")
                print(f"        https://console.cloud.google.com/security/modelarmor")
            elif f['status'] == 429:
                print(f"      • Rate-limit. Re-run with --request_delay 0.5 (or higher).")
            elif isinstance(f['status'], int) and 500 <= f['status'] < 600:
                print(f"      • Transient Model Armor backend error. Retry later.")
            else:
                print(f"      • Inspect the response body above for an explicit reason field.")

    if output_csv:
        try:
            fieldnames = [
                "Row", "Prompt", "Expected_Attack", "Expected_Categories",
                "Detected", "Evaluation", "Matched_Filters", "Skipped_Filters",
                "Latency_ms", "Error_Status",
            ]
            with open(output_csv, 'w', encoding='utf-8', newline='') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
            print(f"\n💾 Detailed per-prompt results saved to: {output_csv}")
        except Exception as e:
            print(f"\n⚠️  Could not write output CSV to {output_csv}: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Armor F1 Score Tester")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--template_id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--request_delay", type=float, default=0.0,
                        help="Delay (s) between requests to avoid rate limiting.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (default: 1).")
    parser.add_argument("--output_csv", default=None,
                        help="Optional file path to export per-prompt evaluation results as CSV.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-attempt HTTP status / exception details for every request.")
    parser.add_argument("--no_category_report", action="store_true",
                        help="Suppress the per-category performance section (binary report only).")

    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    client = ModelArmorClient(
        args.project_id, args.template_id, args.location,
        verbose=args.verbose, max_workers=args.concurrency,
    )
    run_csv_test(
        client, args.csv_file,
        request_delay=args.request_delay,
        category_report=not args.no_category_report,
        concurrency=args.concurrency,
        output_csv=args.output_csv,
    )
