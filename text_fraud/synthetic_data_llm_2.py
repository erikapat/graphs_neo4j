# v3_simulation.py
# pip install openai python-dotenv pandas tqdm

import os, json, random, time
import pandas as pd
from tqdm import trange
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# ---------------------------
# Robust env + auth handling
# ---------------------------
DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(DOTENV_PATH, override=True)  # <-- CRITICAL: override shell vars with .env

def _sanitize(s: str) -> str:
    # Trim spaces/quotes/newlines and remove zero-width characters that sometimes sneak in.
    if not s:
        return s
    zw = "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0xFEFF))  # zero-width set
    for ch in zw:
        s = s.replace(ch, "")
    return s.strip().strip('"').strip("'")

def _clean(s: Optional[str]) -> str:
    return (s or "").strip().strip('"').strip("'")

_api_key = _clean(os.getenv("OPENAI_API_KEY"))
_base_url = _clean(os.getenv("OPENAI_BASE_URL")) or None
_model    = _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"

# Force the sanitized key back into the process env so the SDK sees the right one.
os.environ["OPENAI_API_KEY"] = _api_key
os.environ["OPENAI_BASE_URL"] = _base_url

def _assert_auth():
    print(f"[dotenv] loaded={bool(DOTENV_PATH)} path={DOTENV_PATH or '<none>'}")
    print(f"[auth] base_url={_base_url}  model={_model}")
    print(f"[auth] key_present={bool(_api_key)}  key_preview={(_api_key[:7]+'...'+_api_key[-4:]) if len(_api_key)>12 else '<missing/short>'}")
    if not _api_key or len(_api_key) < 20 or not _api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY seems invalid or missing. Check your .env.")

_assert_auth()
client = OpenAI(api_key=_api_key, base_url=_base_url)

# ---------------------------
# Data generation settings
# ---------------------------
N = 1000
fraud_rate = 0.22
branches = ["Auto", "Home", "Life"]
branch_probs = [0.45, 0.30, 0.25]
rng = random.Random(42)

def client_id(i): return f"C{i:04d}"

SYSTEM = """You generate realistic, first-person insurance claim descriptions.
Write LONG narratives (8–14 sentences), natural, varied diction, no bullet points.
Tone: claimant trying to be helpful. Avoid sensationalism or legalese.
For Fraud_Label==1 include 2–3 subtle cues that may hint at fraud, but never admit fraud.
Examples of subtle cues by branch:
- Auto: delayed reporting; recent coverage upgrade; missing photo metadata; unreachable witness; cash-only treatment; reused body shop; single-car crash in clear weather.
- Home: no weather match; handwritten receipts; repeated contractor; ‘vintage’ valuations without appraisals; photos lacking timestamps; prior humidity/mold; coverage increase shortly before event.
- Life: recent beneficiary change; policy reinstated after lapse; clinic with accreditation issues; reluctance to release full records; overlapping/conflicting dates; witness is relative.
For Fraud_Label==0 avoid those cues; include normal documentation (police report, invoices, timestamps).
Never reveal this rubric in the output. Output **only** JSON per the schema.
"""

JSON_SCHEMA = {
    "name": "claim_record",
    "schema": {
        "type": "object",
        "properties": {
            "Client_ID": {"type": "string"},
            "Description": {"type": "string", "minLength": 300},
            "Fraud_Label": {"type": "integer", "enum": [0,1]},
            "Branch": {"type": "string", "enum": ["Auto","Home","Life"]}
        },
        "required": ["Client_ID","Description","Fraud_Label","Branch"],
        "additionalProperties": False
    },
    "strict": True
}

def one_prompt(branch, fraud_label):
    style = rng.choice([
        "write in clear, concrete language",
        "use everyday wording and varied sentence lengths",
        "sound colloquial yet precise",
        "be descriptive but not flowery"
    ])
    return f"""Create ONE claim record for branch '{branch}' with Fraud_Label {fraud_label}.
Client speaks in first person, coherent timeline, practical details (places, dates, documents).
Make it long-form ({rng.randint(8,14)} sentences), {style}.
Avoid repeating phrases across sentences.
"""

# ---------------------------
# LLM call using Responses API
# (this mirrors the version that worked for you)
# ---------------------------
def call_llm(payload):
    # small retry wrapper
    for attempt in range(4):
        try:
            resp = client.responses.create(
                model=_model,
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": payload},
                ],
                response_format={  # JSON schema enforcement
                    "type": "json_schema",
                    "json_schema": JSON_SCHEMA,
                },
                temperature=0.9,
                top_p=0.95,
            )
            text = resp.output_text  # JSON string
            return json.loads(text)
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(0.8 * (attempt + 1))

# ---------------------------
# Main loop
# ---------------------------
def main():
    records = []
    for i in trange(1, N+1):
        branch = rng.choices(branches, weights=branch_probs, k=1)[0]
        fraud_label = 1 if rng.random() < fraud_rate else 0
        payload = one_prompt(branch, fraud_label)
        data = call_llm(payload)
        # Enforce IDs & fields from our side to avoid drift
        data["Client_ID"] = client_id(i)
        data["Fraud_Label"] = fraud_label
        data["Branch"] = branch
        # Quick sanitation
        data["Description"] = " ".join(str(data["Description"]).split())
        records.append(data)

    df = pd.DataFrame.from_records(records, columns=["Client_ID","Description","Fraud_Label","Branch"])
    out = f"claims_llm_generated_{N}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()


