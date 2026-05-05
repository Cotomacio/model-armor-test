#!/usr/bin/env python3
"""
Model Armor CSV Tester with Intelligent Category Grouping (F1 Score)
------------------------------------------------------------------
This script tests prompts and calculates F1 Scores by grouping 
similar threats (e.g., Harassment, Hate Speech) into their official 
Model Armor filter families (e.g., RESPONSIBLE_AI).

Usage:
  python model_armor_csv_f1.py --project_id YOUR_PROJECT --template_id YOUR_TEMPLATE --location us-central1 --csv_file dados.csv
"""

import csv
import argparse
import sys
import time
from collections import defaultdict
from typing import List, Dict, Any, Set

import requests
import google.auth
from google.auth.transport.requests import Request

class ModelArmorClient:
    def __init__(self, project_id: str, template_id: str, location: str):
        self.project_id = project_id
        self.template_id = template_id
        self.location = location
        self.access_token = None

    def _get_access_token(self):
        try:
            credentials, _ = google.auth.default()
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            import subprocess
            try:
                res = subprocess.run(
                    ["/Users/cotomacio/google-cloud-sdk/bin/gcloud", "auth", "application-default", "print-access-token"],
                    capture_output=True, text=True, check=True
                )
                token = res.stdout.strip()
                if token:
                    return token
            except Exception as err:
                print(f"[FALLBACK ERROR] Failed to run gcloud: {err}")
            raise e

    def sanitize(self, prompt: str) -> Dict[str, Any]:
        try:
            if not self.access_token:
                self.access_token = self._get_access_token()
            
            url = f"https://modelarmor.{self.location}.rep.googleapis.com/v1alpha/projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}:sanitizeUserPrompt"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            payload = {"user_prompt_data": {"text": prompt}}
            
            for attempt in range(2):
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 401:
                    self.access_token = self._get_access_token()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    continue
                break
            
            return response.json()
        except Exception as e:
            return {"error": str(e), "is_error": True}


def map_category(raw_category: str) -> str:
    """
    Funil de Categorias: Mapeia rótulos do CSV e da API para 
    os grandes grupos oficiais do Model Armor.
    """
    cat = raw_category.upper().strip()
    
    # Agrupamento Responsible AI (RAI)
    if cat in ('TOXICITY', 'HATE_SPEECH', 'HARASSMENT', 'DANGEROUS', 'SEXUALLY_EXPLICIT', 'RAI', 'RESPONSIBLE_AI'):
        return 'RESPONSIBLE_AI'
        
    # Agrupamento Injeção de Prompt
    if cat in ('PROMPT_INJECTION', 'JAILBREAK', 'PI_AND_JAILBREAK'):
        return 'PROMPT_INJECTION'
        
    # Agrupamento Dados Sensíveis
    if cat in ('PII', 'SDP', 'SENSITIVE_DATA'):
        return 'PII'
        
    # Agrupamento URLs Maliciosas
    if cat in ('MALICIOUS_URIS', 'MALICIOUS_URI', 'URIS'):
        return 'MALICIOUS_URIS'
        
    if cat == 'CSAM':
        return 'CSAM'
        
    # Retorna a categoria original caso não caia em nenhuma regra (fallback)
    return cat


def extract_triggered_categories(response_data: Dict[str, Any]) -> Set[str]:
    """Extrai e mapeia os filtros disparados pela API para os grupos oficiais."""
    triggered = set() # Usamos um Set (conjunto) para evitar duplicatas (ex: 2 filtros RAI disparados contam como 1)
    sanitization = response_data.get('sanitizationResult', {})
    
    if sanitization.get('filterMatchState') != 'MATCH_FOUND':
        return triggered

    filter_results = sanitization.get('filterResults', {})
    
    # Prompt Injection
    pi_jb = filter_results.get('pi_and_jailbreak', {}).get('piAndJailbreakFilterResult', {})
    if pi_jb.get('matchState') == 'MATCH_FOUND':
        triggered.add(map_category('PROMPT_INJECTION'))
        
    # Responsible AI (Mapeia qualquer sub-filtro para RESPONSIBLE_AI)
    rai = filter_results.get('rai', {}).get('raiFilterResult', {})
    if rai.get('matchState') == 'MATCH_FOUND':
        rai_types = rai.get('raiFilterTypeResults', {})
        for rai_name, rai_data in rai_types.items():
            if rai_data.get('matchState') == 'MATCH_FOUND':
                triggered.add(map_category('RESPONSIBLE_AI'))
                
    # SDP / PII
    sdp = filter_results.get('sdp', {}).get('sdpFilterResult', {}).get('inspectResult', {})
    if sdp.get('matchState') == 'MATCH_FOUND':
        triggered.add(map_category('PII'))
        
    # CSAM
    csam = filter_results.get('csam', {}).get('csamFilterFilterResult', {})
    if csam.get('matchState') == 'MATCH_FOUND':
        triggered.add(map_category('CSAM'))
        
    # Malicious URIs
    uris = filter_results.get('malicious_uris', {}).get('maliciousUriFilterResult', {})
    if uris.get('matchState') == 'MATCH_FOUND':
        triggered.add(map_category('MALICIOUS_URIS'))

    return triggered


def calculate_metrics(tp, fp, fn, tn):
    """Calcula Precisão, Recall e F1 Score."""
    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1, total


def run_csv_test(client: ModelArmorClient, filepath: str):
    prompts_data = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if len(row) < 2: continue
                prompt_text = row[0].strip()
                is_attack_str = row[1].strip().lower()
                is_attack = is_attack_str in ('yes', 'y', 'true', '1')
                
                prompts_data.append({
                    "text": prompt_text, 
                    "is_attack": is_attack
                })
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        sys.exit(1)

    print(f"\n🚀 Iniciando Teste ({len(prompts_data)} prompts)")
    print("-" * 100)
    print(f"{'PROMPT (Preview)':<45} | {'EXPECTED ATTACK':<17} | {'DETECTED':<10} | {'EVALUATION'}")
    print("-" * 100)
    
    g_tp, g_fp, g_fn, g_tn = 0, 0, 0, 0
    
    for p_data in prompts_data:
        prompt = p_data['text']
        is_attack_expected = p_data['is_attack']
        
        response_data = client.sanitize(prompt)
        is_blocked = response_data.get('sanitizationResult', {}).get('filterMatchState') == 'MATCH_FOUND'
        
        # Tracking Geral
        if is_attack_expected and is_blocked:
            eval_str, g_tp = "✅ TP (Bloqueio Correto)", g_tp + 1
        elif not is_attack_expected and is_blocked:
            eval_str, g_fp = "❌ FP (Alarme Falso)", g_fp + 1
        elif is_attack_expected and not is_blocked:
            eval_str, g_fn = "❌ FN (Ataque Passou!)", g_fn + 1
        else:
            eval_str, g_tn = "✅ TN (Permitido Correto)", g_tn + 1
                
        prompt_preview = (prompt.replace('\n', ' ')[:42] + '...') if len(prompt) > 42 else prompt.replace('\n', ' ')
        exp_str = "ATTACK" if is_attack_expected else "SAFE"
        det_str = "BLOCKED" if is_blocked else "PASSED"
        print(f"{prompt_preview:<45} | {exp_str:<17} | {det_str:<10} | {eval_str}")

    print("\n" + "=" * 80)
    print("📊 RELATÓRIO DE AVALIAÇÃO DO MODEL ARMOR")
    print("=" * 80)
    
    # Legenda
    print("\n📖 LEGENDA DE MÉTRICAS (Matriz de Confusão):")
    print("   [TP] True Positive  : Ataque real que foi BLOQUEADO corretamente (Sucesso 🛡️)")
    print("   [TN] True Negative  : Prompt seguro que foi PERMITIDO corretamente (Sucesso ✅)")
    print("   [FP] False Positive : Prompt seguro que foi BLOQUEADO incorretamente (Alarme Falso / Fricção ⚠️)")
    print("   [FN] False Negative : Ataque real que foi PERMITIDO incorretamente (Falha de Segurança 🚨)")
    print("-" * 80)

    g_prec, g_rec, g_f1, g_tot = calculate_metrics(g_tp, g_fp, g_fn, g_tn)
    print(f"\n🛡️  PERFORMANCE GERAL (Qualquer Bloqueio)")
    print(f"   Total de Prompts: {g_tot}")
    print(f"   Métricas Base   : TP={g_tp} | TN={g_tn} | FP={g_fp} | FN={g_fn}")
    print(f"   Precisão        : {g_prec:.4f}")
    print(f"   Recall          : {g_rec:.4f}")
    print(f"   F1 Score Geral  : {g_f1:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Armor F1 Score Tester")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--template_id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--csv_file", required=True)
    
    args = parser.parse_args()
    client = ModelArmorClient(args.project_id, args.template_id, args.location)
    run_csv_test(client, args.csv_file)