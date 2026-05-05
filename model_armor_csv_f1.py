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
  python model_armor_csv_f1.py --project_id YOUR_PROJECT --template_id YOUR_TEMPLATE --location us-central1 --csv_file dados.csv
"""

import csv
import argparse
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, Set, List, Tuple

import requests
import google.auth
from google.auth.transport.requests import Request

# Categorias oficiais agrupadas que o Model Armor pode reportar.
KNOWN_CATEGORIES = ('PROMPT_INJECTION', 'RESPONSIBLE_AI', 'PII', 'MALICIOUS_URIS', 'CSAM')


class ModelArmorClient:
    def __init__(self, project_id: str, template_id: str, location: str):
        self.project_id = project_id
        self.template_id = template_id
        self.location = location
        self._credentials = None
        # Latência de cada chamada HTTP bem-sucedida (em milissegundos).
        self.latencies_ms: List[float] = []

    def _get_credentials(self):
        if self._credentials is None:
            try:
                self._credentials, _ = google.auth.default()
            except Exception as e:
                # Fallback: tenta usar gcloud do PATH (sem caminho hardcoded).
                gcloud = shutil.which("gcloud")
                if not gcloud:
                    raise RuntimeError(
                        "Falha ao obter credenciais ADC e 'gcloud' não foi encontrado no PATH."
                    ) from e
                try:
                    res = subprocess.run(
                        [gcloud, "auth", "application-default", "print-access-token"],
                        capture_output=True, text=True, check=True, timeout=15,
                    )
                    token = res.stdout.strip()
                    if not token:
                        raise RuntimeError("gcloud retornou token vazio.")
                    # Cria um wrapper mínimo que se comporta como Credentials para o resto do fluxo.
                    self._credentials = _StaticTokenCredentials(token)
                except subprocess.SubprocessError as err:
                    raise RuntimeError(f"Falha ao executar gcloud: {err}") from e
        return self._credentials

    def _access_token(self) -> str:
        creds = self._get_credentials()
        # Refresh proativo se o token estiver expirado/prestes a expirar.
        if getattr(creds, "expired", False) or not getattr(creds, "token", None):
            try:
                creds.refresh(Request())
            except Exception:
                # Para credenciais estáticas (fallback gcloud), refresh é no-op.
                pass
        return creds.token

    def sanitize(self, prompt: str) -> Dict[str, Any]:
        url = (
            f"https://modelarmor.{self.location}.rep.googleapis.com/v1alpha/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"templates/{self.template_id}:sanitizeUserPrompt"
        )
        payload = {"user_prompt_data": {"text": prompt}}

        backoff = 1.0
        last_response = None
        for attempt in range(4):
            try:
                token = self._access_token()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                start = time.perf_counter()
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                latency_ms = (time.perf_counter() - start) * 1000.0
                last_response = response

                if response.status_code == 401:
                    # Token rejeitado: força refresh no próximo loop.
                    self._credentials = None
                    continue
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue

                response.raise_for_status()
                self.latencies_ms.append(latency_ms)
                return response.json()
            except requests.RequestException:
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue

        status = last_response.status_code if last_response is not None else "no_response"
        return {"is_error": True, "error_status": status}


class _StaticTokenCredentials:
    """Wrapper mínimo para um token estático obtido via subprocess `gcloud`."""

    def __init__(self, token: str):
        self.token = token
        self.expired = False

    def refresh(self, request):  # noqa: ARG002
        # No-op: tokens estáticos não podem ser renovados aqui.
        return None


def map_category(raw_category: str) -> str:
    """Mapeia rótulos do CSV e da API para os grandes grupos oficiais do Model Armor."""
    cat = raw_category.upper().strip()

    if cat in ('TOXICITY', 'HATE_SPEECH', 'HARASSMENT', 'DANGEROUS', 'SEXUALLY_EXPLICIT', 'RAI', 'RESPONSIBLE_AI'):
        return 'RESPONSIBLE_AI'
    if cat in ('PROMPT_INJECTION', 'JAILBREAK', 'PI_AND_JAILBREAK'):
        return 'PROMPT_INJECTION'
    if cat in ('PII', 'SDP', 'SENSITIVE_DATA'):
        return 'PII'
    if cat in ('MALICIOUS_URIS', 'MALICIOUS_URI', 'URIS'):
        return 'MALICIOUS_URIS'
    if cat == 'CSAM':
        return 'CSAM'
    return cat


def extract_triggered_categories(response_data: Dict[str, Any]) -> Set[str]:
    """Extrai e mapeia os filtros disparados pela API para os grupos oficiais."""
    triggered: Set[str] = set()
    sanitization = response_data.get('sanitizationResult', {})

    if sanitization.get('filterMatchState') != 'MATCH_FOUND':
        return triggered

    filter_results = sanitization.get('filterResults', {})

    pi_jb = filter_results.get('pi_and_jailbreak', {}).get('piAndJailbreakFilterResult', {})
    if pi_jb.get('matchState') == 'MATCH_FOUND':
        triggered.add('PROMPT_INJECTION')

    rai = filter_results.get('rai', {}).get('raiFilterResult', {})
    if rai.get('matchState') == 'MATCH_FOUND':
        rai_types = rai.get('raiFilterTypeResults', {})
        for _, rai_data in rai_types.items():
            if rai_data.get('matchState') == 'MATCH_FOUND':
                triggered.add('RESPONSIBLE_AI')
                break

    sdp = filter_results.get('sdp', {}).get('sdpFilterResult', {}).get('inspectResult', {})
    if sdp.get('matchState') == 'MATCH_FOUND':
        triggered.add('PII')

    csam = filter_results.get('csam', {}).get('csamFilterFilterResult', {})
    if csam.get('matchState') == 'MATCH_FOUND':
        triggered.add('CSAM')

    uris = filter_results.get('malicious_uris', {}).get('maliciousUriFilterResult', {})
    if uris.get('matchState') == 'MATCH_FOUND':
        triggered.add('MALICIOUS_URIS')

    return triggered


def percentile(values: List[float], p: float) -> float:
    """Percentil com interpolação linear (mesmo método do numpy default)."""
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
    """Calcula Precisão, Recall e F1 Score."""
    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, total


def parse_expected_categories(raw: str) -> Set[str]:
    """Parseia a coluna Category do CSV (suporta múltiplas categorias separadas por '|' ou ',')."""
    if not raw:
        return set()
    parts = [p for chunk in raw.split('|') for p in chunk.split(',')]
    return {map_category(p) for p in parts if p.strip()}


def run_csv_test(client: ModelArmorClient, filepath: str, request_delay: float = 0.0):
    prompts_data: List[Dict[str, Any]] = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            has_category = bool(header) and len(header) >= 3 and header[2].strip().lower().startswith('categor')

            for row in reader:
                if len(row) < 2:
                    continue
                prompt_text = row[0].strip()
                is_attack_str = row[1].strip().lower()
                is_attack = is_attack_str in ('yes', 'y', 'true', '1')
                expected_cats: Set[str] = set()
                if has_category and len(row) >= 3:
                    expected_cats = parse_expected_categories(row[2].strip())

                prompts_data.append({
                    "text": prompt_text,
                    "is_attack": is_attack,
                    "expected_categories": expected_cats,
                })
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        sys.exit(1)

    print(f"\n🚀 Iniciando Teste ({len(prompts_data)} prompts)")
    print("-" * 110)
    print(f"{'PROMPT (Preview)':<45} | {'EXPECTED':<17} | {'DETECTED':<10} | {'EVALUATION'}")
    print("-" * 110)

    g_tp = g_fp = g_fn = g_tn = 0
    g_errors = 0
    # Confusão por categoria: { 'PROMPT_INJECTION': {tp,fp,fn,tn}, ... }
    cat_stats: Dict[str, Dict[str, int]] = {
        c: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for c in KNOWN_CATEGORIES
    }

    for p_data in prompts_data:
        prompt = p_data['text']
        is_attack_expected = p_data['is_attack']
        expected_cats: Set[str] = p_data['expected_categories']

        response_data = client.sanitize(prompt)

        if response_data.get('is_error'):
            g_errors += 1
            eval_str = f"⚠️  ERROR ({response_data.get('error_status')})"
            exp_str = "ATTACK" if is_attack_expected else "SAFE"
            det_str = "—"
            prompt_preview = (prompt.replace('\n', ' ')[:42] + '...') if len(prompt) > 42 else prompt.replace('\n', ' ')
            print(f"{prompt_preview:<45} | {exp_str:<17} | {det_str:<10} | {eval_str}")
            if request_delay:
                time.sleep(request_delay)
            continue

        is_blocked = response_data.get('sanitizationResult', {}).get('filterMatchState') == 'MATCH_FOUND'
        triggered_cats = extract_triggered_categories(response_data)

        # Métricas binárias gerais.
        if is_attack_expected and is_blocked:
            eval_str = "✅ TP (Bloqueio Correto)"
            g_tp += 1
        elif not is_attack_expected and is_blocked:
            eval_str = "❌ FP (Alarme Falso)"
            g_fp += 1
        elif is_attack_expected and not is_blocked:
            eval_str = "❌ FN (Ataque Passou!)"
            g_fn += 1
        else:
            eval_str = "✅ TN (Permitido Correto)"
            g_tn += 1

        # Métricas por categoria (cada categoria é um classificador binário independente).
        for cat in KNOWN_CATEGORIES:
            expected = cat in expected_cats
            predicted = cat in triggered_cats
            if expected and predicted:
                cat_stats[cat]['tp'] += 1
            elif not expected and predicted:
                cat_stats[cat]['fp'] += 1
            elif expected and not predicted:
                cat_stats[cat]['fn'] += 1
            else:
                cat_stats[cat]['tn'] += 1

        prompt_preview = (prompt.replace('\n', ' ')[:42] + '...') if len(prompt) > 42 else prompt.replace('\n', ' ')
        exp_str = "ATTACK" if is_attack_expected else "SAFE"
        det_str = "BLOCKED" if is_blocked else "PASSED"
        print(f"{prompt_preview:<45} | {exp_str:<17} | {det_str:<10} | {eval_str}")

        if request_delay:
            time.sleep(request_delay)

    print("\n" + "=" * 80)
    print("📊 RELATÓRIO DE AVALIAÇÃO DO MODEL ARMOR")
    print("=" * 80)

    print("\n📖 LEGENDA DE MÉTRICAS (Matriz de Confusão):")
    print("   [TP] True Positive  : Ataque real que foi BLOQUEADO corretamente (Sucesso 🛡️)")
    print("   [TN] True Negative  : Prompt seguro que foi PERMITIDO corretamente (Sucesso ✅)")
    print("   [FP] False Positive : Prompt seguro que foi BLOQUEADO incorretamente (Alarme Falso / Fricção ⚠️)")
    print("   [FN] False Negative : Ataque real que foi PERMITIDO incorretamente (Falha de Segurança 🚨)")
    print("-" * 80)

    g_prec, g_rec, g_f1, g_tot = calculate_metrics(g_tp, g_fp, g_fn, g_tn)
    print(f"\n🛡️  PERFORMANCE GERAL (Qualquer Bloqueio)")
    print(f"   Total de Prompts Avaliados : {g_tot}")
    print(f"   Erros de API (excluídos)   : {g_errors}")
    print(f"   Métricas Base   : TP={g_tp} | TN={g_tn} | FP={g_fp} | FN={g_fn}")
    print(f"   Precisão        : {g_prec:.4f}")
    print(f"   Recall          : {g_rec:.4f}")
    print(f"   F1 Score Geral  : {g_f1:.4f}")

    # F1 por categoria (somente se o CSV trouxer rótulos de categoria).
    has_any_expected = any(p['expected_categories'] for p in prompts_data)
    if has_any_expected:
        print("\n" + "-" * 80)
        print("📂 PERFORMANCE POR CATEGORIA")
        print("-" * 80)
        print(f"{'CATEGORIA':<20} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | {'PREC':>6} | {'REC':>6} | {'F1':>6}")
        f1_values = []
        for cat in KNOWN_CATEGORIES:
            s = cat_stats[cat]
            prec, rec, f1, _ = calculate_metrics(s['tp'], s['fp'], s['fn'], s['tn'])
            # Só inclui no macro-F1 categorias com algum exemplo positivo esperado ou predito.
            if s['tp'] + s['fp'] + s['fn'] > 0:
                f1_values.append(f1)
            print(f"{cat:<20} | {s['tp']:>4} | {s['fp']:>4} | {s['fn']:>4} | {s['tn']:>4} | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f}")

        if f1_values:
            macro_f1 = sum(f1_values) / len(f1_values)
            print(f"\n   F1 Macro (média entre categorias com sinal): {macro_f1:.4f}")
    else:
        print("\n(ℹ️  Coluna 'Category' ausente ou vazia no CSV — F1 por categoria não calculado.)")

    # Relatório de latência (apenas chamadas HTTP bem-sucedidas).
    print("\n" + "-" * 80)
    print("⏱️  LATÊNCIA DAS CHAMADAS À API DO MODEL ARMOR")
    print("-" * 80)
    samples = client.latencies_ms
    if not samples:
        print("(Sem amostras de latência — nenhuma chamada bem-sucedida.)")
    else:
        n = len(samples)
        mean_ms = sum(samples) / n
        print(f"   Amostras (HTTP 2xx): {n}")
        print(f"   min  : {min(samples):>8.1f} ms")
        print(f"   mean : {mean_ms:>8.1f} ms")
        print(f"   p50  : {percentile(samples, 50):>8.1f} ms")
        print(f"   p90  : {percentile(samples, 90):>8.1f} ms")
        print(f"   p95  : {percentile(samples, 95):>8.1f} ms")
        print(f"   p99  : {percentile(samples, 99):>8.1f} ms")
        print(f"   max  : {max(samples):>8.1f} ms")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Armor F1 Score Tester")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--template_id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--request_delay", type=float, default=0.0,
                        help="Atraso (s) entre requisições para evitar rate limiting.")

    args = parser.parse_args()
    client = ModelArmorClient(args.project_id, args.template_id, args.location)
    run_csv_test(client, args.csv_file, request_delay=args.request_delay)
