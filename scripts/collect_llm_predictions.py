"""
Script to collect predictions from a single LLM with robust error handling and resumable collection.

This script:
1. Loads dataset (train, val, and/or test splits)
2. Runs inference with specified LLM multiple times
3. Saves predictions to a cache file for later ensemble analysis
4. Handles truncation errors gracefully with detection and logging
5. Saves intermediate results periodically to prevent data loss
6. Supports resuming interrupted collections
7. Supports expanding n_runs without re-collecting existing runs
8. Supports parallel processing for faster collection

Supports:
- Azure OpenAI
- Standard OpenAI
- Claude (Anthropic)
- Google Gemini (via Google AI Studio)
- Biomni (via agenecy API)
- Local LLMs (via OpenAI-compatible API)
- GNEsys

Features:
- Parallel Processing: Use num_workers to process multiple examples concurrently
- Truncation Detection: Automatically detects when LM responses are truncated and flags them
- Intermediate Saves: Saves results every N examples (configurable via save_frequency)
- Resume Support: If interrupted, re-run with same config to continue where you left off
- Expand Runs: Increase n_runs (e.g., 5 -> 10) to add more runs without re-collecting
- Retry Failed: Use retry_failed=true to re-run examples where all predictions failed (empty)

Usage:
  # Collect from Azure GPT-4
  python scripts/collect_llm_predictions.py model_name=gpt4 lm.provider=azure lm.model=gpt-4

  # Collect from Claude
  python scripts/collect_llm_predictions.py model_name=claude lm.provider=anthropic lm.model=claude-3-5-sonnet-20241022
  
  # Collect from Biomni
  python scripts/collect_llm_predictions.py model_name=biomni-a1-claude-4 lm.provider=biomni lm.model=biomni-a1-claude-4
  
  # Collect from local LLM
  python scripts/collect_llm_predictions.py model_name=llama3 lm.provider=local lm.model=llama-3.1-70b
  
  # Expand existing collection (5 runs -> 10 runs, only collects 5 more)
  python scripts/collect_llm_predictions.py model_name=llama3 collection.n_runs=10
  
  # Use parallel processing with 8 workers
  python scripts/collect_llm_predictions.py model_name=llama3 collection.num_workers=8
  
  # Adjust save frequency
  python scripts/collect_llm_predictions.py model_name=llama3 collection.save_frequency=10
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

# Load environment variables from .env file
import dotenv
dotenv.load_dotenv('.env', override=True)

import os
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any, Tuple
import numpy as np
import warnings
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
import requests
import urllib3
import dspy
from dspy import Example


from screensqa.dataset.dataset import BioGRIDDSPY
from promptopt.utils.gnesys_wrapper import GNEsysPredictor, GNEsysLM

# Set up logging to catch dspy warnings
logging.basicConfig(level=logging.WARNING)


class ServerHealthMonitor:
    """
    Monitors vLLM server health via response latency and error rates.
    Triggers a server restart (via PID file from serve_with_restart.sh)
    when degradation is detected.
    
    Detection signals:
    - Consecutive API failures (connection errors, timeouts)
    - Response latency: when recent median latency exceeds slowdown_factor x
      the baseline, the server is stuck in thinking loops and needs a restart.
    
    The latency approach catches the common case where the server doesn't fail
    but gets progressively slower as requests get stuck in reasoning loops.
    """
    
    def __init__(self, api_base: str, failure_threshold: int = 5,
                 health_poll_interval: float = 10.0, max_wait: float = 600.0,
                 slowdown_factor: float = 3.0, latency_window: int = 10,
                 warmup_requests: int = 5):
        self.api_base = api_base.rstrip('/')
        self.health_url = self.api_base.replace('/v1', '') + '/health'
        self.failure_threshold = failure_threshold
        self.health_poll_interval = health_poll_interval
        self.max_wait = max_wait
        self.slowdown_factor = slowdown_factor
        self.latency_window = latency_window
        self.warmup_requests = warmup_requests
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._recovering = threading.Event()
        self._recovering.set()  # set = healthy, clear = recovering
        self._restart_lock = threading.Lock()
        
        # Latency tracking
        self._baseline_latencies = []
        self._recent_latencies = []
        self._baseline_median = None
        self._total_requests = 0
        self._last_restart_time = 0.0  # requests started before this are stale
        
        import re
        port_match = re.search(r':(\d+)', api_base)
        self.port = port_match.group(1) if port_match else '8000'
        self.pid_file = f"/tmp/vllm-server-{self.port}.pid"
    
    def record_success(self, latency: float = None, req_start: float = None):
        with self._lock:
            self._consecutive_failures = 0
            if latency is None or req_start is None:
                return
            # Discard latencies from requests that started before the last
            # restart — they include restart wait time and would poison the
            # baseline.
            if req_start < self._last_restart_time:
                return
            self._total_requests += 1
            if self._total_requests <= self.warmup_requests:
                self._baseline_latencies.append(latency)
                if len(self._baseline_latencies) == self.warmup_requests:
                    sorted_bl = sorted(self._baseline_latencies)
                    self._baseline_median = sorted_bl[len(sorted_bl) // 2]
                    print(f"  [health] Baseline latency: {self._baseline_median:.1f}s "
                          f"(from first {self.warmup_requests} requests)", flush=True)
            else:
                self._recent_latencies.append(latency)
                if len(self._recent_latencies) > self.latency_window:
                    self._recent_latencies.pop(0)
    
    def _check_latency_degradation(self) -> bool:
        """Check if recent latencies indicate degradation. Must hold lock."""
        if (self._baseline_median is None or
                len(self._recent_latencies) < self.latency_window):
            return False
        sorted_recent = sorted(self._recent_latencies)
        recent_median = sorted_recent[len(sorted_recent) // 2]
        threshold = self._baseline_median * self.slowdown_factor
        if recent_median > threshold:
            print(f"\n📈 Latency degradation: recent median {recent_median:.1f}s "
                  f"vs baseline {self._baseline_median:.1f}s "
                  f"(>{self.slowdown_factor}x)", flush=True)
            return True
        return False
    
    def check_and_trigger_restart(self) -> bool:
        """Check if we should trigger a restart based on latency. Called after each success."""
        with self._lock:
            if not self._recovering.is_set():
                return False
            if self._check_latency_degradation():
                self._recovering.clear()
                return True
            return False
    
    def record_failure(self) -> bool:
        """Record a failure. Returns True if we've hit the threshold and recovery is starting."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold and self._recovering.is_set():
                self._recovering.clear()
                return True
            return False
    
    def _kill_server(self):
        """Kill the vLLM server via PID file so the restart wrapper can relaunch it."""
        import signal
        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"🔪 Sent SIGTERM to vLLM server (PID {pid})", flush=True)
        except FileNotFoundError:
            print(f"⚠️  PID file {self.pid_file} not found — is serve_with_restart.sh running?", flush=True)
            print(f"   Waiting for manual restart...", flush=True)
        except (ProcessLookupError, ValueError) as e:
            print(f"⚠️  Could not kill server: {e}", flush=True)
    
    def _wait_for_server(self):
        """Wait for the server to come back up after restart."""
        time.sleep(5)
        elapsed = 0.0
        while elapsed < self.max_wait:
            try:
                resp = requests.get(self.health_url, timeout=5)
                if resp.status_code == 200:
                    time.sleep(5)
                    print(f"✅ Server restarted after {elapsed:.0f}s, resuming collection...",
                          flush=True)
                    return
            except Exception:
                pass
            time.sleep(self.health_poll_interval)
            elapsed += self.health_poll_interval
        print(f"⚠️  Server did not recover after {self.max_wait:.0f}s, resuming anyway...",
              flush=True)
    
    def wait_if_recovering(self):
        """
        Block until the server is healthy again.  Called by workers before
        each request AND after failures.
        
        Exactly one thread acquires _restart_lock and performs the kill +
        wait cycle.  All other threads block on _recovering.wait() until
        that thread signals completion.
        """
        if self._recovering.is_set():
            return
        
        acquired = self._restart_lock.acquire(blocking=False)
        if acquired:
            try:
                print(f"\n⏸️  Restarting server to clear degraded state...", flush=True)
                self._kill_server()
                self._wait_for_server()
                with self._lock:
                    self._consecutive_failures = 0
                    self._recent_latencies.clear()
                    self._baseline_latencies.clear()
                    self._baseline_median = None
                    self._total_requests = 0
                    self._last_restart_time = time.time()
                self._recovering.set()
            finally:
                self._restart_lock.release()
        else:
            self._recovering.wait(timeout=self.max_wait)


# Global health monitor, set during collection setup
_server_health_monitor: ServerHealthMonitor = None


def create_ranking_signature(system_prompt: str = None):
    """
    Create a RankingSignature class with the given system prompt as docstring.
    """
    if system_prompt is None:
        system_prompt = "Signature for gene ranking task."
    
    class RankingSignature(dspy.Signature):
        __doc__ = system_prompt
        
        question = dspy.InputField(desc="The gene ranking task description")
        answer = dspy.OutputField(desc="Comma-separated list of genes in ranked order")
    
    return RankingSignature


class RankingModule(dspy.Module):
    """DSPy module for gene ranking."""
    
    def __init__(self, signature_class=None):
        super().__init__()
        if signature_class is None:
            signature_class = create_ranking_signature()
        self.predictor = dspy.ChainOfThought(signature_class)
    
    def forward(self, question: str) -> dspy.Prediction:
        result = self.predictor(question=question)
        return result


def create_dspy_examples(dataset_examples: List[Dict[str, Any]]) -> List[Example]:
    """Convert dataset examples to DSPy Example format.
    
    Supports both ScreensQADSPY and BioGRIDDSPY dataset formats.
    """
    dspy_examples = []
    
    for ex in dataset_examples:
        dspy_ex = dspy.Example(
            question=ex['question'],
            answer=ex.get('answer', '')
        ).with_inputs("question")
        
        # Handle both 'genes' (ScreensQADSPY) and 'relevance_genes' (BioGRIDDSPY)
        dspy_ex.genes = ex.get('genes', ex.get('relevance_genes', []))
        
        # Store metadata - common fields
        for key in ['relevance_scores', 'dataset_name', 'phenotype', 'num_genes']:
            if key in ex:
                setattr(dspy_ex, key, ex[key])
        
        # ScreensQADSPY-specific fields
        for key in ['alpha', 'ranking_method', 'description', 'split', 'reverse']:
            if key in ex:
                setattr(dspy_ex, key, ex[key])
        
        # BioGRIDDSPY-specific fields
        for key in ['hit', 'cell_type', 'cell_line', 'screen_type', 
                    'library_methodology', 'screen_rationale']:
            if key in ex:
                setattr(dspy_ex, key, ex[key])
        
        dspy_examples.append(dspy_ex)
    
    return dspy_examples


def load_additional_split_examples(
    split_name: str,
    paths: List[str],
    use_existing_prompt: bool = False,
    display_library_genes: bool = False,
) -> List[Example]:
    """
    Load examples from additional HuggingFace datasets and return DSPy Examples.

    For biogrid-compatible datasets (use_existing_prompt=False), generates prompts
    using the biogrid_ranking_prompt template.  For datasets that already contain
    a ``prompt`` column (use_existing_prompt=True), uses that column directly.
    """
    from datasets import load_from_disk
    from screensqa.utils.prompt_loaders import load_objective_prompt
    from screensqa.data_generation.biogrid_generation import _extract_screen_ids_from_dataset_name

    prompt_template = None
    if not use_existing_prompt:
        prompt_template = load_objective_prompt("biogrid_ranking_prompt")

    all_examples: List[Dict[str, Any]] = []

    for ds_path in paths:
        ds = load_from_disk(ds_path)
        print(f"    Loaded {len(ds)} examples from {ds_path}")

        for item in ds:
            if use_existing_prompt:
                prompt = item['prompt']
            else:
                phenotype = item.get('phenotype', '')
                if phenotype and phenotype[-1] == '.':
                    item = dict(item)
                    item['phenotype'] = phenotype[:-1]
                prompt = prompt_template.format(**item)

            if display_library_genes and len(item['relevance_genes']) < 2000:
                prompt += (
                    "\n\nOnly the following genes were considered in this screen. "
                    "Only output genes from this list: "
                    + ', '.join(item['relevance_genes'])
                )

            top10_args = np.argsort(item['relevance_scores'])[::-1]
            top10_genes = np.array(item['relevance_genes'])[top10_args][:10].tolist()

            example = {
                'question': prompt,
                'relevance_genes': item['relevance_genes'],
                'relevance_scores': item['relevance_scores'],
                'hit': item['hit'],
                'dataset_name': item.get('dataset_name', 'unknown'),
                'phenotype': item.get('phenotype', 'Not specified'),
                'num_genes': len(item['relevance_genes']),
                'answer': ', '.join(top10_genes),
            }
            all_examples.append(example)

    print(f"  {split_name}: {len(all_examples)} total examples from {len(paths)} dataset(s)")
    return create_dspy_examples(all_examples)


import re

# Compiled pattern for validating HGNC-like gene symbols:
# Starts with uppercase letter, followed by uppercase letters/digits/hyphens, 2-15 chars total.
# e.g., TP53, BRCA1, HLA-A, CDKN2A, C1orf43
_GENE_SYMBOL_RE = re.compile(r'^[A-Z][A-Z0-9][-A-Z0-9]{0,13}$')

# Blocklist of common uppercase tokens that pass the gene regex but are NOT genes.
# These are database column names, metadata labels, file format tokens, etc. that
# can appear in agentic model output (e.g., DepMap column headers, format keywords).
_NON_GENE_BLOCKLIST = frozenset({
    # DepMap / database column name fragments
    'ID', 'RRID', 'CCLE', 'CCLEN', 'COSMICID', 'BROAD', 'WTSIM',
    'TCGACODE', 'DEPMAP', 'SANGER', 'ENTREZ',
    # Common abbreviations / format tokens
    'CSV', 'TSV', 'JSON', 'HTML', 'HTTP', 'HTTPS', 'URL', 'URI',
    'PDF', 'PNG', 'JPG', 'GIF', 'SVG',
    'NULL', 'NONE', 'TRUE', 'FALSE', 'NAN', 'INF',
    'AND', 'NOT', 'THE', 'FOR', 'WITH', 'FROM', 'INTO',
    'GENE', 'GENES', 'SYMBOL', 'NAME', 'TYPE', 'CODE',
    'FDR', 'PVALUE', 'LOG', 'MEAN', 'STD', 'VAR',
    'RNA', 'DNA', 'MRNA', 'CDNA',  # biological terms, not gene symbols
    'CRISPR', 'CRISPRI', 'CRISPRN', 'KNOCKOUT', 'KO',
    'MODEL', 'DATA', 'INDEX', 'ROW', 'COL', 'COLUMN',
    'FILE', 'PATH', 'DIR',
})


def parse_genes_from_output(output_text: str) -> List[str]:
    """Parse gene list from model output.
    
    Splits by comma and validates each token looks like an HGNC gene symbol.
    Tokens that are too long, contain lowercase words, or otherwise don't
    match the gene pattern are filtered out.  Also filters out known
    non-gene tokens (database column names, format keywords, etc.).
    """
    # Handle None or empty output (e.g., from truncated responses)
    if output_text is None or not isinstance(output_text, str):
        return []
    
    genes = []
    for token in output_text.split(','):
        token = token.strip()
        if not token:
            continue
        # If the token is a clean gene symbol, keep it directly
        if _GENE_SYMBOL_RE.match(token):
            if token not in _NON_GENE_BLOCKLIST:
                genes.append(token)
        else:
            # Try to extract a gene symbol from within the token
            # (handles cases like "PAFAH1B1\end{solution}..." or "1. TP53")
            m = _GENE_SYMBOL_RE.search(token) if len(token) < 50 else None
            if m and m.group() not in _NON_GENE_BLOCKLIST:
                genes.append(m.group())
            # Otherwise skip -- not a gene symbol
    return genes

def extract_genes_from_raw_response(raw_text: str) -> List[str]:
    """
    Extract gene list from raw LM response text, handling agentic model output
    that includes markdown, code blocks, tool output, etc.
    
    Strategy order (highest to lowest confidence):
      0. Parse the "answer" field from a \\begin{solution} JSON block (biomni-specific)
      1. Find the longest comma-separated gene list line in the response
      2. Find numbered lists (1. TP53, 2. BRCA1, ...)
      3. Collect all gene-like tokens from the entire response
    
    Args:
        raw_text: Raw response text from the LM
        
    Returns:
        List of gene symbols, or empty list if none found
    """
    if not raw_text or not isinstance(raw_text, str):
        return []
    
    gene_re = re.compile(r'[A-Z][A-Z0-9][-A-Z0-9]{0,13}')
    
    def _extract_gene(token: str) -> str:
        """Extract a gene symbol from a token, filtering out blocklisted terms."""
        m = gene_re.search(token)
        if m and m.group() not in _NON_GENE_BLOCKLIST:
            return m.group()
        return None
    
    # Strategy 0: Extract the "answer" field from \begin{solution}...\end{solution}
    # Biomni wraps its final answer in this block as JSON with "answer" key.
    # This is the most authoritative source -- use it if available.
    solution_match = re.search(
        r'\\begin\{solution\}\s*(.*?)\s*\\end\{solution\}',
        raw_text,
        re.DOTALL
    )
    if solution_match:
        solution_text = solution_match.group(1).strip()
        # Try to parse as JSON and extract the "answer" field
        try:
            solution_json = json.loads(solution_text)
            if isinstance(solution_json, dict) and 'answer' in solution_json:
                answer_text = solution_json['answer']
                tokens = [t.strip() for t in answer_text.split(',')]
                gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
                if len(gene_tokens) >= 5:
                    return gene_tokens
        except (json.JSONDecodeError, ValueError, TypeError):
            # JSON parse failed (e.g., truncated answer). Try regex extraction
            # from the answer field value directly.
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)', solution_text)
            if answer_match:
                answer_text = answer_match.group(1)
                tokens = [t.strip() for t in answer_text.split(',')]
                gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
                if len(gene_tokens) >= 5:
                    return gene_tokens
    
    # Strategy 1: Find lines that look like comma-separated gene lists
    # (at least 5 comma-separated gene-like tokens on one line)
    # Uses search within each token to handle leading/trailing noise
    best_genes = []
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('```'):
            continue
        tokens = [t.strip() for t in line.split(',')]
        gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
        if len(gene_tokens) >= 5 and len(gene_tokens) > len(best_genes):
            best_genes = gene_tokens
    
    if best_genes:
        return best_genes
    
    # Strategy 2: Find numbered lists like "1. TP53\n2. BRCA1\n..."
    numbered_raw = re.findall(r'^\s*\d+[\.\)]\s*([A-Z][A-Z0-9][-A-Z0-9]{0,13})\b', raw_text, re.MULTILINE)
    numbered = [g for g in numbered_raw if g not in _NON_GENE_BLOCKLIST]
    if len(numbered) >= 5:
        return numbered
    
    # Strategy 3: Collect all gene-like tokens from the entire response
    gene_pattern = r'\b[A-Z][A-Z0-9][-A-Z0-9]{0,13}\b'
    all_genes = re.findall(gene_pattern, raw_text)
    seen = set()
    unique_genes = []
    for g in all_genes:
        if g not in seen and len(g) >= 2 and g not in _NON_GENE_BLOCKLIST:
            seen.add(g)
            unique_genes.append(g)
    if len(unique_genes) >= 5:
        return unique_genes
    
    return []


def check_for_truncation(prediction, lm) -> bool:
    """
    Check if the last prediction was truncated.
    Returns True if truncation was detected.
    """
    try:
        # Check dspy history for truncation indicators
        history = lm.history
        if history and len(history) > 0:
            last_call = history[-1]
            # Check if response has finish_reason indicating truncation
            if hasattr(last_call, 'response'):
                response = last_call['response']
                if isinstance(response, dict):
                    finish_reason = response.get('choices', [{}])[0].get('finish_reason', '')
                    if finish_reason == 'length':
                        return True
                elif hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'finish_reason'):
                        if response.choices[0].finish_reason == 'length':
                            return True
    except Exception as e:
        # If we can't check, assume no truncation
        pass
    
    return False


def load_existing_predictions(output_file: Path) -> Dict:
    """
    Load existing predictions from file if it exists.
    
    Returns:
        Dictionary with existing predictions or empty dict if file doesn't exist
    """
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"  Loaded existing predictions from {output_file}")
                print(f"  Found {data.get('num_examples', 0)} examples with {data.get('n_runs', 0)} runs each")
                return data
        except Exception as e:
            print(f"  Warning: Could not load existing predictions: {e}")
    return {}


def merge_predictions(existing: List[Dict], new_question: str, new_preds: List, new_truncated: List, new_reasoning: List = None) -> Dict:
    """
    Merge new predictions with existing ones for a question.
    
    Returns:
        Merged prediction dictionary
    """
    # Find existing entry for this question
    existing_entry = None
    for entry in existing:
        if entry.get('question') == new_question:
            existing_entry = entry
            break
    
    if existing_entry is None:
        # No existing predictions for this question
        result = {
            'question': new_question,
            'predictions': new_preds,
            'truncated': new_truncated,
            'any_truncated': any(new_truncated)
        }
        if new_reasoning is not None:
            result['reasoning'] = new_reasoning
        return result
    else:
        # Merge with existing predictions
        merged_preds = existing_entry.get('predictions', []) + new_preds
        merged_truncated = existing_entry.get('truncated', []) + new_truncated
        result = {
            'question': new_question,
            'predictions': merged_preds,
            'truncated': merged_truncated,
            'any_truncated': any(merged_truncated)
        }
        if new_reasoning is not None:
            existing_reasoning = existing_entry.get('reasoning', [None] * len(existing_entry.get('predictions', [])))
            result['reasoning'] = existing_reasoning + new_reasoning
        return result


def save_intermediate_results(output_file: Path, cache_data: Dict):
    """Save intermediate results to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def process_single_example(
    model: 'RankingModule',
    example: Example,
    example_idx: int,
    n_runs: int,
    existing_entry: Dict = None,
    verbose: bool = True
) -> Tuple[int, Dict, int]:
    """
    Process a single example with all its runs.
    
    Args:
        model: The ranking module
        example: The example to process
        example_idx: Index of the example (for logging)
        n_runs: Target number of runs
        existing_entry: Existing predictions for this example (if any)
        verbose: Print progress
    
    Returns:
        Tuple of (example_idx, result_entry, truncation_count)
    """
    truncation_count = 0
    
    # Determine how many runs we already have
    if existing_entry:
        existing_preds = existing_entry.get('predictions', [])
        existing_truncated = existing_entry.get('truncated', [])
        existing_reasoning = existing_entry.get('reasoning', [None] * len(existing_preds))
        runs_completed = len(existing_preds)
        
        if runs_completed >= n_runs:
            # Already have enough runs, return existing
            return (example_idx, existing_entry, 0)
        else:
            # Need to collect more runs
            predictions = existing_preds.copy()
            truncation_flags = existing_truncated.copy()
            reasoning_list = existing_reasoning.copy()
            start_run = runs_completed
    else:
        # No existing predictions
        predictions = []
        truncation_flags = []
        reasoning_list = []
        start_run = 0
    
    for run_idx in range(start_run, n_runs):
        # Wait if server is recovering from a crash
        if _server_health_monitor is not None:
            _server_health_monitor.wait_if_recovering()
        
        # Add suffix to prompt
        question_with_suffix = example.question + f"\n\nYour goal is to provide a list of genes that meet the screen criteria, even if you do not have access to the actual experimental data. The genes must use HGNC symbols. Use your knowledge of biology, gene function, and relevant pathways to predict which genes are most likely to be hits. Do not refuse to answer or say you need more data—make your best predictions based on your understanding of the biological context."

        try:
            req_start = time.time()
            prediction = model(question=question_with_suffix)
            req_latency = time.time() - req_start
            output_text = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
            genes = parse_genes_from_output(output_text)
            
            # Extract raw response text (used for reasoning and fallback extraction)
            raw_response_text = None
            try:
                lm = dspy.settings.lm
                if hasattr(lm, 'history') and lm.history:
                    last_resp = lm.history[-1]
                    if isinstance(last_resp, dict):
                        raw = last_resp.get('response', {})
                        if isinstance(raw, dict):
                            raw_response_text = raw.get('content', None)
                        elif isinstance(raw, str):
                            raw_response_text = raw
            except Exception:
                pass
            
            # Quality gate: if the happy path extracted very few genes,
            # the extraction likely went wrong (e.g., thread-safety issue
            # with lm.history, or DSPy parsed garbage as the answer).
            # Try the more robust fallback extraction from raw response.
            MIN_EXPECTED_GENES = 20
            if len(genes) < MIN_EXPECTED_GENES and raw_response_text:
                fallback_genes = extract_genes_from_raw_response(raw_response_text)
                if len(fallback_genes) > len(genes):
                    if verbose:
                        print(f"  [Ex {example_idx+1}] Run {run_idx+1}: quality gate triggered "
                              f"({len(genes)} genes from DSPy parse, "
                              f"re-extracted {len(fallback_genes)} from raw response)", flush=True)
                    genes = fallback_genes
            
            # Extract reasoning if available (from DSPy ChainOfThought)
            reasoning_text = None
            if hasattr(prediction, 'reasoning') and prediction.reasoning:
                reasoning_text = prediction.reasoning
            elif raw_response_text:
                # Use raw LM response as reasoning (useful for agentic models
                # like biomni where the full response contains tool output)
                reasoning_text = raw_response_text
            
            # Check for truncation
            was_truncated = check_for_truncation(prediction, dspy.settings.lm)
            
            predictions.append(genes)
            truncation_flags.append(was_truncated)
            reasoning_list.append(reasoning_text)
            
            if was_truncated:
                truncation_count += 1
            
            if _server_health_monitor is not None:
                _server_health_monitor.record_success(latency=req_latency, req_start=req_start)
                if _server_health_monitor.check_and_trigger_restart():
                    print(f"\n🔴 Latency degradation detected, restarting server...", flush=True)
                    _server_health_monitor.wait_if_recovering()
            
            if verbose:
                print(f"  [Ex {example_idx+1}] Run {run_idx+1}: {len(genes)} genes{' [TRUNCATED]' if was_truncated else ''}", flush=True)
                
        except Exception as e:
            error_msg = str(e)
            
            is_server_error = any(kw in error_msg.lower() for kw in [
                'connection error', 'timeout', 'timed out', 'connection refused',
                'internalservererror', 'bad gateway', '502', '503',
            ])
            if is_server_error and _server_health_monitor is not None:
                triggered = _server_health_monitor.record_failure()
                if triggered:
                    print(f"\n🔴 Server degradation detected, pausing workers...", flush=True)
                _server_health_monitor.wait_if_recovering()
            
            # Check if this is a DSPy parsing error (response was received but
            # couldn't be parsed into structured fields). This is common with
            # agentic models like biomni that include markdown/tool output.
            is_parse_error = (
                'cannot be serialized' in error_msg or
                'JSONAdapter' in error_msg or
                'Failed to parse' in error_msg
            )
            
            fallback_genes = []
            raw_text = None
            if is_parse_error:
                # Try to extract genes from the raw LM response
                try:
                    lm = dspy.settings.lm
                    if hasattr(lm, 'history') and lm.history:
                        last_response = lm.history[-1]
                        # BiomniLM stores {'content': text} in response
                        if isinstance(last_response, dict):
                            raw_text = last_response.get('response', {})
                            if isinstance(raw_text, dict):
                                raw_text = raw_text.get('content', '')
                            elif isinstance(raw_text, str):
                                pass
                            else:
                                raw_text = str(raw_text)
                        else:
                            raw_text = str(last_response)
                        fallback_genes = extract_genes_from_raw_response(raw_text)
                except Exception:
                    pass
            
            if fallback_genes:
                print(f"  [Ex {example_idx+1}] Run {run_idx+1}: {len(fallback_genes)} genes (extracted from raw response)", flush=True)
                predictions.append(fallback_genes)
                truncation_flags.append(False)
                # Store the full raw response as reasoning for fallback extractions
                reasoning_list.append(raw_text)
            else:
                print(f"  ❌ [Ex {example_idx+1}] ERROR run {run_idx+1}: {error_msg[:500]}", flush=True)
                
                # Check if error is truncation-related
                is_truncation_error = (
                    'NoneType' in error_msg or 
                    'truncat' in error_msg.lower()
                )
                
                if is_truncation_error:
                    truncation_count += 1
                
                # Store empty prediction on error
                predictions.append([])
                truncation_flags.append(is_truncation_error)
                reasoning_list.append(None)
    
    result_entry = {
        'question': example.question,
        'predictions': predictions,
        'truncated': truncation_flags,
        'any_truncated': any(truncation_flags),
        'reasoning': reasoning_list,
    }
    
    return (example_idx, result_entry, truncation_count)


def has_failed_predictions(entry: Dict) -> bool:
    """
    Check if an entry has any failed predictions (empty lists).
    
    Returns:
        True if any prediction is empty (failed), False otherwise
    """
    predictions = entry.get('predictions', [])
    if not predictions:
        return True
    return any(len(pred) == 0 for pred in predictions)


def filter_failed_predictions(entry: Dict) -> Dict:
    """
    Return a copy of the entry with failed (empty) predictions removed,
    so that process_single_example can fill in replacement runs.
    """
    predictions = entry.get('predictions', [])
    truncated = entry.get('truncated', [False] * len(predictions))
    reasoning = entry.get('reasoning', [None] * len(predictions))
    
    filtered = {
        'question': entry['question'],
        'predictions': [],
        'truncated': [],
        'reasoning': [],
    }
    for pred, trunc, reason in zip(predictions, truncated, reasoning):
        if len(pred) > 0:
            filtered['predictions'].append(pred)
            filtered['truncated'].append(trunc)
            filtered['reasoning'].append(reason)
    
    filtered['any_truncated'] = any(filtered['truncated'])
    return filtered


def collect_predictions(
    model: RankingModule,
    examples: List[Example],
    n_runs: int,
    split_name: str,
    output_file: Path = None,
    model_name: str = None,
    lm_config: Dict = None,
    save_frequency: int = 5,
    num_workers: int = 1,
    retry_failed: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Collect predictions from model with support for resuming, expanding runs, and parallel processing.
    
    Args:
        model: The ranking module to use for predictions
        examples: List of examples to process
        n_runs: Target number of runs per example
        split_name: Name of the split (train/val/test)
        output_file: Path to save intermediate/final results
        model_name: Name of the model for metadata
        lm_config: LM configuration for metadata
        save_frequency: Save intermediate results every N examples
        num_workers: Number of parallel workers (default: 1 for sequential)
        retry_failed: If True, re-run examples where all predictions failed (empty)
        verbose: Print progress messages
    
    Returns:
        List of prediction dictionaries with question, predictions, and truncation info
    """
    # Load existing predictions if available
    existing_data = {}
    existing_predictions = []
    if output_file and output_file.exists():
        existing_data = load_existing_predictions(output_file)
        existing_predictions = existing_data.get('predictions', [])
    
    # Build lookup for existing predictions by question
    existing_by_question = {entry['question']: entry for entry in existing_predictions}
    
    # Initialize results storage (indexed by example position to maintain order)
    results_by_idx = {}
    total_truncations = 0
    examples_completed = 0
    
    # Thread-safe lock for updating shared state
    lock = threading.Lock()
    
    def save_progress():
        """Save intermediate results (must be called with lock held)."""
        if not output_file:
            return
        
        # Build ordered list of predictions
        all_preds = []
        for idx in range(len(examples)):
            if idx in results_by_idx:
                all_preds.append(results_by_idx[idx])
            elif examples[idx].question in existing_by_question:
                all_preds.append(existing_by_question[examples[idx].question])
        
        num_truncated = sum(1 for p in all_preds if p.get('any_truncated', False))
        total_truncs = sum(sum(p.get('truncated', [])) for p in all_preds)
        
        cache_data = {
            'model_name': model_name or 'unknown',
            'lm_config': lm_config or {},
            'n_runs': n_runs,
            'split': split_name,
            'num_examples': len(all_preds),
            'num_examples_with_truncation': num_truncated,
            'total_truncated_runs': total_truncs,
            'predictions': all_preds,
            'status': 'in_progress',
            'examples_processed': examples_completed,
            'total_examples': len(examples)
        }
        
        save_intermediate_results(output_file, cache_data)
    
    # Prepare tasks - each task is (example_idx, example, existing_entry)
    tasks = []
    skipped = 0
    retried = 0
    for i, example in enumerate(examples):
        existing_entry = existing_by_question.get(example.question)
        if existing_entry and len(existing_entry.get('predictions', [])) >= n_runs:
            # Check if we should retry failed predictions
            if retry_failed and has_failed_predictions(existing_entry):
                filtered = filter_failed_predictions(existing_entry)
                tasks.append((i, example, filtered))
                retried += 1
            else:
                # Already complete, add to results directly
                results_by_idx[i] = existing_entry
                skipped += 1
        else:
            tasks.append((i, example, existing_entry))
    
    if verbose:
        status_msg = f"\n{split_name}: {len(examples)} total examples, {skipped} already complete, {len(tasks)} to process"
        if retried > 0:
            status_msg += f" ({retried} retrying failed)"
        print(status_msg, flush=True)
        if num_workers > 1:
            print(f"  Using {num_workers} parallel workers", flush=True)
    
    if len(tasks) == 0:
        # All examples already complete
        return [results_by_idx[i] for i in range(len(examples))]
    
    # Process examples (parallel or sequential)
    if num_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for (idx, example, existing_entry) in tasks:
                future = executor.submit(
                    process_single_example,
                    model, example, idx, n_runs, existing_entry, verbose
                )
                future_to_idx[future] = idx
            
            # Process completed futures
            for future in as_completed(future_to_idx):
                try:
                    example_idx, result_entry, truncs = future.result()
                    
                    with lock:
                        results_by_idx[example_idx] = result_entry
                        total_truncations += truncs
                        examples_completed += 1
                        
                        # Save periodically
                        if examples_completed % save_frequency == 0:
                            if verbose:
                                print(f"\n💾 Saving progress ({examples_completed}/{len(tasks)} new examples)...", flush=True)
                            save_progress()
                            
                except Exception as e:
                    print(f"  ❌ Task failed: {e}", flush=True)
    else:
        # Sequential processing (original behavior)
        for (idx, example, existing_entry) in tasks:
            if verbose:
                print(f"\n{split_name} - Processing example {idx+1}/{len(examples)}...", flush=True)
            
            example_idx, result_entry, truncs = process_single_example(
                model, example, idx, n_runs, existing_entry, verbose
            )
            
            results_by_idx[example_idx] = result_entry
            total_truncations += truncs
            examples_completed += 1
            
            # Save periodically
            if output_file and (examples_completed % save_frequency == 0):
                if verbose:
                    print(f"\n💾 Saving progress ({examples_completed}/{len(tasks)} new examples)...", flush=True)
                save_progress()
    
    # Build final ordered list
    all_predictions = []
    for i in range(len(examples)):
        if i in results_by_idx:
            all_predictions.append(results_by_idx[i])
        elif examples[i].question in existing_by_question:
            all_predictions.append(existing_by_question[examples[i].question])
    
    if total_truncations > 0:
        print(f"\n⚠️  WARNING: {total_truncations} truncations detected across all predictions!", flush=True)
        print(f"   Consider increasing max_tokens further or reviewing long responses.", flush=True)
    
    return all_predictions


class BiomniLM(dspy.LM):
    """
    Custom DSPy LM for Biomni via agenecy API.
    
    Uses streaming (stream=true) to avoid nginx proxy timeouts on long-running
    agentic requests. The biomni model runs tools and databases under the hood,
    which can take minutes -- without streaming the connection gets closed by
    nginx before the response is ready (empty body / 502).
    
    Uses requests directly with verify=False to bypass SSL certificate issues.
    Includes retry logic with exponential backoff for transient failures.
    """
    
    def __init__(self, api_key: str, api_base: str, model: str,
                 max_tokens: int = 16000, temperature: float = 1.0,
                 max_retries: int = 10, initial_backoff: float = 5.0, **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.history = []
        self.url = f"{api_base}/chat/completions"
        # Suppress InsecureRequestWarning from urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def _stream_request(self, headers, payload):
        """
        Make a streaming API request and accumulate the full response text
        from Server-Sent Events (SSE) chunks.
        
        Streaming keeps the connection alive through nginx proxy timeouts
        because the server sends incremental chunks as the model generates.
        
        Returns:
            The complete response text assembled from all streamed chunks.
        """
        payload_with_stream = {**payload, "stream": True}
        
        with requests.Session() as session:
            session.verify = False
            resp = session.post(
                self.url,
                headers=headers,
                json=payload_with_stream,
                stream=True,
                timeout=(30, 600),  # (connect_timeout, read_timeout)
            )
        
            # Check HTTP status
            if resp.status_code != 200:
                body = resp.text[:300] if resp.text else "(empty body)"
                raise ValueError(f"HTTP {resp.status_code}: {body}")
            
            # Read SSE stream and accumulate content
            content_parts = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # SSE format: "data: {...}" or "data: [DONE]"
                if line.startswith("data: "):
                    data_str = line[6:]  # strip "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Extract delta content from the chunk
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                content_parts.append(content)
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue
        
        if not content_parts:
            raise ValueError("Stream completed but no content was received")
        
        return "".join(content_parts)
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Call Biomni API with streaming and retry logic.
        
        Args:
            prompt: String prompt (alternative to messages)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of response strings
        """
        import time
        import random
        
        if messages is None:
            if prompt is not None:
                messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError("Either prompt or messages must be provided")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response_text = self._stream_request(headers, payload)
                
                # Store in history for truncation checking
                self.history.append({
                    'prompt': prompt,
                    'messages': messages,
                    'response': {'content': response_text},
                })
                
                return [response_text]
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff with jitter, capped at 60s
                    backoff = min(self.initial_backoff * (2 ** attempt) + random.uniform(0, 2), 60)
                    print(f"    [BiomniLM] Attempt {attempt+1}/{self.max_retries+1} failed: {str(e)[:200]}. Retrying in {backoff:.1f}s...", flush=True)
                    time.sleep(backoff)
        
        # All retries exhausted
        raise RuntimeError(f"BiomniLM failed after {self.max_retries + 1} attempts. Last error: {last_error}")
    
    def inspect_history(self, n: int = 1):
        """Inspect the last n calls."""
        return self.history[-n:]


@hydra.main(version_base="1.3", config_path="../configs", config_name="collect-predictions")
def main(cfg: DictConfig):
    """
    Main function for collecting LLM predictions.
    """
    print("="*60)
    print("COLLECT LLM PREDICTIONS")
    print("="*60)
    
    # Get model name for organizing outputs
    model_name = cfg.get('model_name', 'unnamed_model')
    print(f"\nModel: {model_name}")
    print(f"Runs per example: {cfg.collection.n_runs}")
    
    # Set random seed
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Initialize LM based on configuration
    print(f"\nInitializing LM ({cfg.lm.provider})...")
    
    if cfg.lm.provider == 'gnesys':
        # GNEsys configuration
        print("  Initializing GNEsys...")
        gnesys_predictor = GNEsysPredictor.from_config_path(
            config_path=cfg.gnesys.config_path,
            config_name=cfg.gnesys.config_name,
            overrides=cfg.gnesys.overrides,
            verbose=cfg.gnesys.verbose
        )
        
        lm = GNEsysLM(
            gnesys_predictor=gnesys_predictor,
            reuse_kernels=cfg.gnesys.get('reuse_kernels', True),
            max_kernels=cfg.gnesys.get('max_kernels', 5)
        )
        
        init_prompt = gnesys_predictor.get_init_prompt()
        RankingSignatureClass = create_ranking_signature(init_prompt)
        
    elif cfg.lm.provider == 'azure':
        # Azure OpenAI configuration
        # Support both AZURE_API_KEY and AZURE_OPENAI_API_KEY
        api_key = os.environ.get('AZURE_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        # Support both AZURE_API_BASE and AZURE_OPENAI_ENDPOINT
        api_base = os.environ.get('AZURE_API_BASE') or os.environ.get('AZURE_OPENAI_ENDPOINT')
        api_version = os.environ.get('AZURE_API_VERSION', '2025-04-01-preview')
        
        if not api_key:
            raise ValueError("Azure OpenAI requires AZURE_API_KEY or AZURE_OPENAI_API_KEY in .env")
        
        if api_base:
            # Ensure AZURE_API_BASE is set for litellm internals
            os.environ['AZURE_API_BASE'] = api_base
            print(f"  Using Azure OpenAI: {api_base}")
        else:
            print(f"  Using Azure OpenAI (endpoint from litellm defaults)")
        print(f"  Deployment: {cfg.lm.model}")
        
        lm_kwargs = dict(
            model=f"azure/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
            api_version=api_version,
        )
        if api_base:
            lm_kwargs['api_base'] = api_base
        lm = dspy.LM(**lm_kwargs)
        
        RankingSignatureClass = create_ranking_signature()
        
    elif cfg.lm.provider == 'anthropic':
        # Claude/Anthropic configuration
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError("Anthropic requires ANTHROPIC_API_KEY in .env")
        
        print(f"  Using Anthropic Claude")
        print(f"  Model: {cfg.lm.model}")
        
        lm = dspy.LM(
            model=f"anthropic/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
        
        RankingSignatureClass = create_ranking_signature()
        
    elif cfg.lm.provider == 'gemini':
        # Google Gemini configuration (via Google AI Studio)
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Google Gemini requires GEMINI_API_KEY in .env or environment")
        
        print(f"  Using Google Gemini")
        print(f"  Model: {cfg.lm.model}")
        
        lm = dspy.LM(
            model=f"gemini/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
        
        RankingSignatureClass = create_ranking_signature()
        
    elif cfg.lm.provider == 'portkey':
        # Portkey gateway configuration (for Galileo/Roche AI Gateway)
        from portkey_ai import createHeaders
        
        api_key = cfg.lm.get('api_key') or os.environ.get('PORTKEY_API_KEY')
        api_base = cfg.lm.get('api_base') or os.environ.get('PORTKEY_API_BASE', '')
        portkey_provider = cfg.lm.get('portkey_provider', 'bedrock')
        
        if not api_key:
            raise ValueError("Portkey requires api_key in config or PORTKEY_API_KEY in .env")
        
        print(f"  Using Portkey Gateway: {api_base}")
        print(f"  Provider: {portkey_provider}")
        print(f"  Model: {cfg.lm.model}")
        
        # Create Portkey headers
        portkey_headers = createHeaders(
            api_key=api_key,
            provider=portkey_provider,
        )
        
        # Note: Portkey/Bedrock requires max_completion_tokens instead of max_tokens
        # Pass via model_kwargs for DSPy compatibility
        lm = dspy.LM(
            model=f"openai/{cfg.lm.model}",
            api_base=api_base,
            api_key="placeholder",  # Portkey ignores this, auth is via headers
            extra_headers=portkey_headers,
            temperature=cfg.lm.get('temperature', 1.0),
            model_kwargs={'max_completion_tokens': cfg.lm.get('max_tokens', 8000)},
        )
        
        RankingSignatureClass = create_ranking_signature()
        
    elif cfg.lm.provider == 'biomni':
        # Biomni configuration (via agenecy OpenAI-compatible API)
        # Uses custom BiomniLM class with requests + verify=False to bypass SSL issues
        api_key = os.environ.get('AGENECY_API_KEY')
        api_base = cfg.lm.get('api_base') or os.environ.get('BIOMNI_API_BASE', '')
        
        if not api_key:
            raise ValueError("Biomni requires AGENECY_API_KEY in .env")
        
        print(f"  Using Biomni (agenecy)")
        print(f"  Model: {cfg.lm.model}")
        
        lm = BiomniLM(
            api_key=api_key,
            api_base=api_base,
            model=cfg.lm.model,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
        
        RankingSignatureClass = create_ranking_signature()
        
    elif cfg.lm.provider == 'local':
        # Local LLM via OpenAI-compatible API
        api_base = cfg.lm.get('api_base', 'http://localhost:8000/v1')
        api_key = cfg.lm.get('api_key', 'token-abc123')

        print(f"  Using Local LLM: {api_base}")
        print(f"  Model: {cfg.lm.model}")
        
        # Set up server health monitor for auto-recovery
        global _server_health_monitor
        _server_health_monitor = ServerHealthMonitor(
            api_base=api_base,
            failure_threshold=cfg.lm.get('failure_threshold', 5),
            slowdown_factor=cfg.lm.get('slowdown_factor', 3.0),
            latency_window=cfg.lm.get('latency_window', 10),
            warmup_requests=cfg.lm.get('warmup_requests', 5),
        )
        print(f"  Health monitor: failure_threshold={_server_health_monitor.failure_threshold}, "
              f"slowdown_factor={_server_health_monitor.slowdown_factor}x")
        print(f"    PID file: {_server_health_monitor.pid_file}")
        print(f"    Health URL: {_server_health_monitor.health_url}")
        
        # Build extra kwargs for sampling parameters
        extra_kwargs = {}
        if cfg.lm.get('top_p') is not None:
            extra_kwargs['top_p'] = cfg.lm.top_p
        if cfg.lm.get('top_k') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['top_k'] = cfg.lm.top_k
        if cfg.lm.get('min_p') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['min_p'] = cfg.lm.min_p
        # Support presence_penalty (standard OpenAI API parameter)
        if cfg.lm.get('presence_penalty') is not None:
            extra_kwargs['presence_penalty'] = cfg.lm.presence_penalty
        # Support repetition_penalty (vllm-specific, passed via extra_body)
        if cfg.lm.get('repetition_penalty') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['repetition_penalty'] = cfg.lm.repetition_penalty
        # Support chat_template_kwargs for thinking mode control (e.g. DeepSeek, GLM)
        if cfg.lm.get('chat_template_kwargs') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['chat_template_kwargs'] = dict(cfg.lm.chat_template_kwargs)
        # Support thinking_budget to cap reasoning tokens separately from output
        if cfg.lm.get('thinking_budget') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['thinking_budget'] = cfg.lm.thinking_budget
        
        lm = dspy.LM(
            model=f"openai/{cfg.lm.model}",
            api_base=api_base,
            api_key=api_key,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
            timeout=cfg.lm.get('timeout', 600),
            num_retries=cfg.lm.get('num_retries', 8),
            **extra_kwargs,
        )
        
        RankingSignatureClass = create_ranking_signature()
        
    else:
        # Standard OpenAI configuration
        print(f"  Using OpenAI")
        print(f"  Model: {cfg.lm.model}")
        
        lm = dspy.LM(
            model=cfg.lm.model,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0)
        )
        
        RankingSignatureClass = create_ranking_signature()
    
    # Configure DSPy
    dspy.configure(lm=lm)
    
    # Disable caching to ensure diverse predictions
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = BioGRIDDSPY(
        dataset_path=cfg.dataset.dataset_path,
        split_type=cfg.dataset.split_type,
        fold=cfg.dataset.fold,
    )
    
    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    
    # Convert to DSPy format
    train_dspy = create_dspy_examples(train_examples) if cfg.collection.collect_train else []
    val_dspy = create_dspy_examples(val_examples) if cfg.collection.collect_val else []
    test_dspy = create_dspy_examples(test_examples) if cfg.collection.collect_test else []
    
    print(f"  Train: {len(train_dspy)} examples (collect: {cfg.collection.collect_train})")
    print(f"  Val:   {len(val_dspy)} examples (collect: {cfg.collection.collect_val})")
    print(f"  Test:  {len(test_dspy)} examples (collect: {cfg.collection.collect_test})")
    
    # Create ranking module
    print("\nCreating ranking module...")
    model = RankingModule(signature_class=RankingSignatureClass)
    
    # Prepare output directory and LM config for metadata
    output_dir = Path(cfg.output.save_dir) / "llm_predictions" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    lm_config = OmegaConf.to_container(cfg.lm, resolve=True)
    
    # Get save frequency from config (default: save every 5 examples)
    save_frequency = cfg.collection.get('save_frequency', 5)
    
    # Get number of parallel workers (default: 1 for sequential processing)
    num_workers = cfg.collection.get('num_workers', 1)
    if num_workers > 1:
        print(f"\n🚀 Parallel processing enabled with {num_workers} workers")
    
    # Get retry_failed option (default: False)
    retry_failed = cfg.collection.get('retry_failed', False)
    if retry_failed:
        print(f"🔄 Retry failed predictions enabled")
    
    # Collect predictions for each split
    all_predictions = {}
    
    if cfg.collection.collect_train and len(train_dspy) > 0:
        print("\n" + "="*60)
        print("COLLECTING TRAIN PREDICTIONS")
        print("="*60)
        train_output_file = output_dir / "train_predictions.json"
        train_predictions = collect_predictions(
            model=model,
            examples=train_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="TRAIN",
            output_file=train_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=retry_failed,
            verbose=True
        )
        all_predictions['train'] = train_predictions
    
    if cfg.collection.collect_val and len(val_dspy) > 0:
        print("\n" + "="*60)
        print("COLLECTING VAL PREDICTIONS")
        print("="*60)
        val_output_file = output_dir / "val_predictions.json"
        val_predictions = collect_predictions(
            model=model,
            examples=val_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="VAL",
            output_file=val_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=retry_failed,
            verbose=True
        )
        all_predictions['val'] = val_predictions
    
    if cfg.collection.collect_test and len(test_dspy) > 0:
        print("\n" + "="*60)
        print("COLLECTING TEST PREDICTIONS")
        print("="*60)
        test_output_file = output_dir / "test_predictions.json"
        test_predictions = collect_predictions(
            model=model,
            examples=test_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="TEST",
            output_file=test_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=retry_failed,
            verbose=True
        )
        all_predictions['test'] = test_predictions
    
    # Collect predictions for additional splits (if configured)
    additional_splits = cfg.get('additional_splits', None)
    if additional_splits:
        for split_name, split_cfg in additional_splits.items():
            split_paths = list(split_cfg.get('paths', []))
            if not split_paths:
                continue
            print("\n" + "="*60)
            print(f"COLLECTING {split_name.upper()} PREDICTIONS")
            print("="*60)
            split_examples = load_additional_split_examples(
                split_name=split_name,
                paths=split_paths,
                use_existing_prompt=split_cfg.get('use_existing_prompt', False),
                display_library_genes=split_cfg.get('display_library_genes', False),
            )
            if len(split_examples) > 0:
                split_output_file = output_dir / f"{split_name}_predictions.json"
                split_predictions = collect_predictions(
                    model=model,
                    examples=split_examples,
                    n_runs=cfg.collection.n_runs,
                    split_name=split_name.upper(),
                    output_file=split_output_file,
                    model_name=model_name,
                    lm_config=lm_config,
                    save_frequency=save_frequency,
                    num_workers=num_workers,
                    retry_failed=retry_failed,
                    verbose=True
                )
                all_predictions[split_name] = split_predictions
    
    # Save final predictions (mark as complete)
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)
    
    for split_name, predictions in all_predictions.items():
        output_file = output_dir / f"{split_name}_predictions.json"
        
        # Calculate truncation statistics
        num_truncated = sum(1 for p in predictions if p.get('any_truncated', False))
        total_truncations = sum(sum(p.get('truncated', [])) for p in predictions)
        
        cache_data = {
            'model_name': model_name,
            'lm_config': lm_config,
            'n_runs': cfg.collection.n_runs,
            'split': split_name,
            'num_examples': len(predictions),
            'num_examples_with_truncation': num_truncated,
            'total_truncated_runs': total_truncations,
            'predictions': predictions,
            'status': 'complete'
        }
        
        with open(output_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"\n✅ Saved {split_name} predictions to {output_file}")
        print(f"   Examples: {len(predictions)}, Runs per example: {cfg.collection.n_runs}")
        if num_truncated > 0:
            print(f"   ⚠️  {num_truncated}/{len(predictions)} examples had truncated responses")
            print(f"   ⚠️  {total_truncations} total truncated runs")
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE!")
    print("="*60)
    print(f"\nPredictions saved to: {output_dir}/")
    print(f"Collected splits: {list(all_predictions.keys())}")


if __name__ == "__main__":
    main()










