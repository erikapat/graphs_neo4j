#!/usr/bin/env python3
from __future__ import annotations
import os, re, sys, json, time, random, textwrap
from typing import Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------- Load .env (robust) -----------------------------
env_path = find_dotenv(usecwd=True)
loaded = load_dotenv(env_path, override=False)
print(f"[dotenv] loaded={loaded} path={env_path or '<none>'}")
print(f"[dotenv] cwd={os.getcwd()}")

def _clean(s: Optional[str]) -> str:
    return (s or "").strip().strip('"').strip("'")

OPENAI_API_KEY  = _clean(os.getenv("OPENAI_API_KEY"))
OPENAI_BASE_URL = _clean(os.getenv("OPENAI_BASE_URL")) or None
OPENAI_MODEL    = _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"

def _mask(key: str) -> str:
    if not key: return "<empty>"
    return key[:7] + "..." + key[-4:] if len(key) > 12 else "<short-key>"

print(f"[auth] base_url={OPENAI_BASE_URL or 'https://api.openai.com/v1'}  model={OPENAI_MODEL}")
print(f"[auth] key_present={bool(OPENAI_API_KEY)}  key={_mask(OPENAI_API_KEY)}")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is missing. Put it in a .env in this folder.\n"
          "Format: OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", file=sys.stderr)
    sys.exit(1)

# ----------------------------- OpenAI client -----------------------------
try:
    from openai import OpenAI
except Exception as e:
    print("Please install the official SDK:  pip install openai", file=sys.stderr)
    raise

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------- Config -----------------------------
N_ROWS       = int(os.getenv("N", "1000"))
FRAUD_RATE   = float(os.getenv("FRAUD_RATE", "0.22"))
BRANCHES     = ["Auto", "Home", "Life"]
BRANCH_PROBS = [0.45, 0.30, 0.25]
OUT_FILE     = os.getenv("OUT", f"claims_llm_generated_{N_ROWS}.csv")
RNG          = random.Random(42)

SYSTEM = """\
You generate realistic, first-person insurance claim descriptions.
Write LONG narratives (8–14 sentences), natural, varied diction, no bullet points.
Tone: claimant trying to be helpful. Avoid sensationalism or legalese.

For Fraud_Label==1 include 2–3 subtle cues that may hint at fraud, but never admit fraud.
Examples of subtle cues by branch:
- Auto: delayed reporting; recent coverage upgrade; missing photo metadata; unreachable witness; cash-only treatment;
        reused body shop; single-car crash in clear weather; wildlife or debris claims that don’t fit local ecology.
- Home: no weather match; handwritten receipts; repeated contractor; “vintage” valuations without appraisals;
        photos lacking timestamps; prior humidity/mold; coverage increase shortly before event.
- Life: recent beneficiary change; policy reinstated after lapse; clinic with accreditation issues;
        reluctance to release full records; overlapping/conflicting dates; witness is relative.

For Fraud_Label==0 avoid those cues; include ordinary, verifiable details (police report number, dated invoices, timestamps).
Geographic plausibility: keep wildlife, road names, and weather consistent with the hinted region.
Do NOT mention the rubric or that the text is synthetic. Output only valid JSON per the caller’s request.
"""

REGION_SEEDS = [
    {"country": "US", "examples": "Ohio, I-71, deer; Arizona, I-10, monsoon; Florida, US-1, palm debris"},
    {"country": "ES", "examples": "Madrid M-30, Valladolid A-62; Seville SE-30; Zaragoza A-2"},
    {"country": "UK", "examples": "M25, A1(M), roe deer; drizzle showers; terraced housing"},
    {"country": "CA", "examples": "Ontario 401, moose; BC Hwy 1; icy shoulder in winter"},
]

def client_id(i: int) -> str:
    return f"C{i:04d}"

def one_prompt(branch: str, fraud_label: int) -> str:
    style = RNG.choice([
        "write in clear, concrete language",
        "use everyday wording with varied sentence lengths",
        "sound colloquial yet precise",
        "be descriptive but not flowery"
    ])
    region = RNG.choice(REGION_SEEDS)
    sentences = RNG.randint(8, 14)
    return f"""\
Create ONE claim record for branch '{branch}' with Fraud_Label {fraud_label}.
Claimant speaks in first person with a coherent timeline and practical details (places, dates, documents).
Make it long-form ({sentences} sentences), {style}.
Keep details plausible for region hints: {region["country"]} – e.g., {region["examples"]}.
Avoid repeating phrases; vary syntax across sentences.
Return only a single JSON object with keys: Client_ID, Description, Fraud_Label, Branch.
"""

RE_JSON = re.compile(r"\{.*\}", re.DOTALL)

def try_parse_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("` \n")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = RE_JSON.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def enforce_schema(d, *, branch: str, fraud_label: int, cid: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        d = {}
    desc = d.get("Description", "")
    if not isinstance(desc, str):
        desc = str(desc or "")
    desc = " ".join(desc.split())
    if len(desc) < 300:
        desc += " " + (" ".join(["The details are provided to help the adjuster verify the timeline and documents."] * 3))
    return {"Client_ID": cid, "Description": desc, "Fraud_Label": int(fraud_label), "Branch": str(branch)}

def call_llm(payload: str, *, branch: str, fraud_label: int, cid: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": payload}]
    last = None
    for attempt in range(6):
        try:
            # try JSON mode first
            try_json = attempt < 3
            kwargs = dict(model=OPENAI_MODEL, messages=messages, temperature=0.9, top_p=0.95)
            if try_json:
                kwargs["response_format"] = {"type": "json_object"}  # ignored by older SDKs (TypeError)
            chat = client.chat.completions.create(**kwargs)
            text = chat.choices[0].message.content
            data = try_parse_json(text) or {}
            return enforce_schema(data, branch=branch, fraud_label=fraud_label, cid=cid)
        except TypeError as e:
            if "response_format" in str(e):
                time.sleep(0.5); continue
            last = e
        except Exception as e:
            last = e
            time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries. Last error: {last!r}")

def preflight() -> None:
    try:
        chat = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are a ping."},
                      {"role": "user", "content": "Reply with OK."}],
            max_tokens=2,
        )
        txt = (chat.choices[0].message.content or "").strip().upper()
        if "OK" not in txt:
            print("[preflight] Unexpected reply, continuing anyway:", repr(txt))
    except Exception as e:
        print("\nAuth check failed.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        if OPENAI_BASE_URL:
            print("If using Azure/OpenRouter, set OPENAI_BASE_URL and OPENAI_MODEL to your deployment name.", file=sys.stderr)
        print("Verify your .env path above and that OPENAI_API_KEY is valid (no quotes/trailing spaces).", file=sys.stderr)
        sys.exit(1)

def main():
    preflight()
    rows = []
    with tqdm(total=N_ROWS, desc="Generating claims") as pbar:
        for i in range(1, N_ROWS + 1):
            branch = random.choices(BRANCHES, weights=BRANCH_PROBS, k=1)[0]
            fraud_label = 1 if random.random() < FRAUD_RATE else 0
            cid = client_id(i)
            payload = one_prompt(branch, fraud_label)
            try:
                data = call_llm(payload, branch=branch, fraud_label=fraud_label, cid=cid)
            except Exception as e:
                data = {
                    "Client_ID": cid,
                    "Description": f"Generation error placeholder. (Reason: {type(e).__name__}) "
                                   f"The claimant explains the event in plain language with dates and documents. "
                                   f"This placeholder exists to preserve row count integrity.",
                    "Fraud_Label": int(fraud_label),
                    "Branch": branch,
                }
            rows.append(data)
            pbar.update(1)
    df = pd.DataFrame.from_records(rows, columns=["Client_ID","Description","Fraud_Label","Branch"])
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} rows to {os.path.abspath(OUT_FILE)}")

if __name__ == "__main__":
    main()

