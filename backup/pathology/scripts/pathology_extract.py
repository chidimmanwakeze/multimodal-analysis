"""
pathology_extract.py

Sends each (BRCA-filtered) pathology report to a locally running Ollama
server and extracts a structured JSON record of clinically relevant fields,
explicitly EXCLUDING anything that would leak the prediction target:
  - molecular subtype / diagnosis classification conclusions
  - ER / PR / HER2 receptor status (these effectively ARE the subtype label
    for breast cancer, so they're excluded even though they're not literally
    the word "subtype" -- see conversation for reasoning)

Assumes an Ollama server is already running and reachable (default
http://localhost:11434) with the target model already pulled.

Usage:
    python pathology_extract.py \
        --input pathology-reports-brca.csv \
        --output results/pathology_json/pathology_extracted.jsonl \
        --model llama3.1:8b \
        --host http://localhost:11434

Resumable: if --output already exists, patient IDs already present are
skipped, so a killed/timed-out job can just be re-run.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Fields the model is allowed to extract, each with an explicit expected
# type/format. Earlier testing showed that without this, the model returns
# inconsistent types for the same field across records (e.g. tumor_grade as
# "G III" in one report, 3 in another; pathologic_stage as a plain string in
# one report, a nested {T,N,M} object in another) -- which breaks anything
# downstream expecting a consistent column type. Being explicit about type
# AND giving a concrete example per field is what fixes this.
SCHEMA = {
    "laterality": "string: one of 'left', 'right', 'bilateral', or null if not stated",
    "specimen_type": "string describing the specimen (e.g. 'core biopsy', 'mastectomy specimen')",
    "procedure": "string describing the surgical procedure performed (e.g. 'partial mastectomy')",
    "histologic_type": "string, the histologic diagnosis (e.g. 'invasive ductal carcinoma')",
    "tumor_size_cm": "a single number (float), the LARGEST tumor dimension in centimeters. "
                      "If multiple tumor foci are reported, use the largest one only. "
                      "Never return a list or a string -- always a plain number, e.g. 2.5",
    "tumor_grade": "a single integer: 1, 2, or 3. Convert Roman numerals or descriptive grades "
                    "(e.g. 'Grade III', 'poorly differentiated') to the matching integer 1-3. "
                    "Never return text -- always a plain integer.",
    "multifocality": "boolean true/false: true if multiple tumor foci are described, else false",
    "lymphovascular_invasion": "boolean true/false: true if lymphovascular invasion is present, "
                                "false if explicitly absent/not identified, null if not mentioned",
    "margin_status": "string summary of margin status (e.g. 'negative', 'positive', "
                      "'close, 2mm from inked margin')",
    "lymph_nodes_examined": "integer: total number of lymph nodes examined, null if not stated",
    "lymph_nodes_positive": "integer: number of lymph nodes positive for tumor, null if not stated",
    "pathologic_stage": "string ONLY: the AJCC pathologic TNM stage as a single plain string "
                         "(e.g. 'pT1c pN0(i-) pMX'). Never return a nested object -- always "
                         "one plain string combining T, N, and M into that single string.",
    "necrosis_present": "boolean true/false, null if not mentioned",
    "in_situ_component": "boolean true/false: true if an in-situ (e.g. DCIS) component is "
                          "present, false if none is mentioned. This is a yes/no flag, NOT a "
                          "description -- never return descriptive text here.",
}
ALLOWED_KEYS = set(SCHEMA.keys())

# Expected Python type per field, used to flag (not silently drop) records
# where the model didn't follow the schema -- e.g. tumor_grade returned as
# "G III" instead of 3, or pathologic_stage returned as a nested {T,N,M}
# object instead of a single string. bool is checked separately from int
# since Python's bool is technically a subclass of int.
EXPECTED_TYPES = {
    "laterality": str,
    "specimen_type": str,
    "procedure": str,
    "histologic_type": str,
    "tumor_size_cm": (int, float),
    "tumor_grade": int,
    "multifocality": bool,
    "lymphovascular_invasion": bool,
    "margin_status": str,
    "lymph_nodes_examined": int,
    "lymph_nodes_positive": int,
    "pathologic_stage": str,
    "necrosis_present": bool,
    "in_situ_component": bool,
}


def validate_types(clean: dict) -> list:
    """Returns a list of human-readable warnings for any field whose value
    doesn't match its expected type. Does not modify or drop anything --
    just flags it for review."""
    warnings = []
    for key, value in clean.items():
        if value is None or key not in EXPECTED_TYPES:
            continue
        expected = EXPECTED_TYPES[key]
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected == (int, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:  # str
            ok = isinstance(value, str)
        if not ok:
            warnings.append(f"{key}: expected {expected}, got {type(value).__name__} "
                             f"({value!r})")
    return warnings


# Defensive regex: if any of these terms show up as a KEY in the model's
# output (regardless of instructions), that key is stripped. This guards
# against the model inventing a key like "er_pr_her2_status" that isn't in
# ALLOWED_KEYS but also isn't obviously named "subtype".
FORBIDDEN_KEY_PATTERN = re.compile(
    r"(subtype|receptor|\ber\b|\bpr\b|her2|hormone|ihc|immunohisto|classification)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = f"""You are a clinical data extraction assistant. You will be given the text \
of a breast cancer pathology report. Extract a JSON object with EXACTLY these keys, each with \
the type/format specified. Follow the type for each field exactly and consistently -- this data \
will be used in a structured table, so type consistency across reports matters more than anything \
else.

{json.dumps(SCHEMA, indent=2)}

Rules:
- If a field is not mentioned in the report, set its value to null (never an empty string).
- Follow the exact type specified for each field (number, integer, boolean, or string). \
Never substitute a different type -- e.g. tumor_grade must always be a plain integer, never \
text like "Grade III"; pathologic_stage must always be a single string, never a nested object.
- Do NOT include any field related to ER status, PR status, HER2 status, hormone receptor \
status, immunohistochemistry (IHC) results, molecular subtype, or any diagnostic classification \
that implies tumor subtype. Omit this information entirely, even if present in the report.
- Do NOT add any keys beyond the ones listed above.
- Output ONLY the JSON object. No explanation, no markdown formatting, no code fences.
"""


def build_prompt(report_text: str) -> str:
    return f"Pathology report:\n\n{report_text}\n\nExtract the JSON object now."


def call_ollama(host: str, model: str, report_text: str, timeout: int = 120,
                 num_predict: int = 500, num_ctx: int = 8192) -> dict:
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": build_prompt(report_text),
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw_text = resp.json()["response"]
    parsed = json.loads(raw_text)  # may raise json.JSONDecodeError
    return parsed


def sanitize(parsed: dict) -> dict:
    """Defensive backstop: drop any key not in ALLOWED_KEYS, or matching a
    forbidden leakage term, regardless of what the model actually returned."""
    clean = {}
    for k, v in parsed.items():
        if k not in ALLOWED_KEYS:
            continue
        if FORBIDDEN_KEY_PATTERN.search(k):
            continue
        clean[k] = v
    # ensure all allowed keys are present, even if the model omitted one
    for k in ALLOWED_KEYS:
        clean.setdefault(k, None)
    return clean


def process_row(row: dict, host: str, model: str, max_retries: int, num_predict: int, num_ctx: int):
    """Runs in a worker thread. Returns (record, error) -- exactly one is None."""
    pid = row["Patient ID"]
    filename = row["patient_filename"]
    text = row["text"]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            parsed = call_ollama(host, model, text, num_predict=num_predict, num_ctx=num_ctx)
            clean = sanitize(parsed)
            warnings = validate_types(clean)
            record = {"Patient ID": pid, "patient_filename": filename, **clean}
            if warnings:
                record["_type_warnings"] = warnings
            return record, None
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)
            time.sleep(2 * attempt)  # backoff

    error = {"Patient ID": pid, "patient_filename": filename, "error": last_err}
    return None, error


def load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["Patient ID"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pathology-reports-brca.csv")
    ap.add_argument("--output", default="results/pathology_json/pathology_extracted.jsonl")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1,
                     help="number of concurrent requests to send to Ollama. "
                          "Must not exceed the server's OLLAMA_NUM_PARALLEL setting, "
                          "or requests beyond that just queue server-side anyway.")
    ap.add_argument("--num-predict", type=int, default=500,
                     help="max tokens to generate per report; caps generation length "
                          "since the fixed 14-field JSON schema doesn't need much more.")
    ap.add_argument("--num-ctx", type=int, default=8192,
                     help="context window per request. IMPORTANT: without this cap, Ollama "
                          "reserves the model's full native context (131072 for Llama 3.1) "
                          "PER PARALLEL SLOT, which can try to allocate 100GB+ of KV cache "
                          "and crash the runner. 8192 covers the longest report in this dataset "
                          "(~4000 words / ~5000+ tokens) plus system prompt and output, with margin.")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = output_path.parent / (output_path.stem + "_errors.jsonl")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} pathology reports.")

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} patients already extracted, skipping those.")
        df = df[~df["Patient ID"].isin(done_ids)].reset_index(drop=True)
    print(f"{len(df)} reports to process this run, with {args.workers} concurrent worker(s).\n")

    n_success, n_fail, n_flagged = 0, 0, 0
    n_done = 0
    t0 = time.time()

    rows = df.to_dict("records")

    with open(output_path, "a") as out_f, open(errors_path, "a") as err_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_row, row, args.host, args.model,
                                 args.max_retries, args.num_predict, args.num_ctx)
                for row in rows
            ]
            for future in as_completed(futures):
                record, error = future.result()
                n_done += 1

                if record is not None:
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    n_success += 1
                    if "_type_warnings" in record:
                        n_flagged += 1
                else:
                    err_f.write(json.dumps(error) + "\n")
                    err_f.flush()
                    n_fail += 1

                if n_done % args.log_every == 0 or n_done == len(rows):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = len(rows) - n_done
                    eta = remaining / rate / 60 if rate > 0 else float("nan")
                    print(f"  {n_done}/{len(rows)}  success={n_success} fail={n_fail} "
                          f"flagged={n_flagged}  ({rate:.2f} reports/sec, ETA {eta:.1f} min)")

    print(f"\nDone. {n_success} succeeded, {n_fail} failed, "
          f"{n_flagged} flagged with type warnings (see _type_warnings field).")
    print(f"Output: {output_path}")
    if n_fail:
        print(f"Failures logged to: {errors_path} (re-run this script to retry them)")


if __name__ == "__main__":
    main()
