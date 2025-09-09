# Generate a new synthetic insurance claims CSV (1,000 rows) with long, highly varied, LLM-style descriptions
import random
import pandas as pd
import numpy as np

random.seed(123)
np.random.seed(123)

N = 1000

def client_id(i):
    return f"C{i:04d}"

branches = ["Auto", "Home", "Life"]
branch_probs = [0.45, 0.30, 0.25]
fraud_rate = 0.22  # keep a non-trivial fraud rate

weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
times_of_day = ["early morning","late morning","around noon","early afternoon","late afternoon","early evening","late evening","past midnight"]
cities = ["Seville","Bilbao","Valencia","Madrid","Barcelona","Zaragoza","Malaga","Valladolid","Alicante","Coruña"]
roads = ["A-1","A-2","A-3","AP-7","M-30","M-40"]
weather = ["heavy rain","light drizzle","strong wind","clear skies","dense fog","hail"]
shops = ["QuickFix Auto","Metro Bodyworks","Rápido Taller","Northside Collision","Coastal Glass","City Dent Pro"]
contractors = ["Fenix Restorations","Iberia Home Care","GreenOak Remodeling","Atlantic CleanUp","SeguroDry","BluePeak Services"]
clinics = ["Riverside Clinic","Centro Salud Norte","Amber Medical","Medicus24","WellSpring Center","Clinica Horizonte"]
first_names = ["Alex","Sam","Jordan","Taylor","Carmen","Diego","Lucia","Marcos","Nora","Pablo","Irene","Rafa","Marta","Sergio","Elena"]
relationships = ["neighbor","cousin","coworker","landlord","roommate","uncle","aunt","brother-in-law","sister-in-law","family friend"]

# Utility helpers
def maybe(s, p=0.5):
    return s if random.random() < p else ""

def rand_phone():
    return f"+34 {random.randint(600,799)} {random.randint(100,999)} {random.randint(100,999)}"

def rand_amount():
    return str(random.randint(300, 8500))

def pick(seq):
    return random.choice(seq)

def sentence_cleanup(sentences):
    return " ".join([s.strip() for s in sentences if s and s.strip()]).replace("  ", " ").strip()

# Generators per branch
def gen_auto_legit():
    name = pick(first_names)
    s = [
        f"I'm reporting a traffic incident that happened on {pick(weekdays)} evening near {pick(cities)} on the {pick(roads)}.",
        f"My car drifted slightly when a {pick(weather)} came through, and another driver clipped my rear bumper while changing lanes.",
        f"I pulled over, took photos, and exchanged details; the other driver’s phone is {rand_phone()} and we both waited for assistance.",
        f"The damage is mostly on the rear quarter and bumper cover; the trunk opens but the alignment looks off.",
        f"I have dashcam footage and the timestamp matches my phone GPS; I can provide both if needed.",
        f"The vehicle was towed to {pick(shops)} and I received an initial estimate of EUR {rand_amount()} including parts and paint.",
        f"I notified the police and obtained a brief report number from the officer at the scene.",
        f"There were no injuries beyond some stiffness the next day; I used over-the-counter treatment and went to work as usual."
    ]
    return sentence_cleanup(s)

def gen_auto_fraud():
    # Subtle cues: delayed reporting, missing metadata, recent coverage change, cash chiropractor, same shop as previous claim, unreachable witness
    name = pick(first_names)
    s = [
        f"I'd like to file a claim for a minor crash I had {random.randint(10,45)} days ago near {pick(cities)}; I didn't report it right away because I thought it was nothing.",
        f"It was a low-speed bump at a light on the {pick(roads)} and I felt some neck stiffness later that week.",
        f"I upgraded to comprehensive coverage about {random.randint(7,21)} days before this because my commute changed and I was advised to be safer.",
        f"I took photos but my phone storage was full and I had to reinstall the camera app, so the pictures don't show location or time data anymore.",
        f"A passerby, my {pick(relationships)}, said they saw the other vehicle, but they’ve been hard to reach since they started a new job.",
        f"I visited {pick(clinics)} a couple of times and paid in cash because their card machine was down; they said they can issue receipts later.",
        f"The car is at {pick(shops)}—I’ve used them before and they prepared an estimate very quickly with a similar format as last time.",
        f"I couldn’t wait for the police due to work and left the scene after exchanging a first name only; I hope that’s okay."
    ]
    return sentence_cleanup(s)

def gen_home_legit():
    s = [
        f"This claim is about water damage we discovered on {pick(weekdays)} morning in our home in {pick(cities)}.",
        f"A supply line under the kitchen sink burst during the night, soaking the lower cabinets and part of the adjacent room.",
        f"We shut off the main valve as soon as we noticed and called a plumber who replaced the cracked hose the same day.",
        f"Photos and a short video were taken right away, and we also have the plumber’s invoice with the part numbers listed.",
        f"The flooring has cupped and there is visible swelling on the toe-kick panels; a musty smell developed after two days.",
        f"{pick(contractors)} provided a drying plan with dehumidifiers and a detailed estimate for materials and labor.",
        f"Our neighbors can confirm when we brought fans inside, and the condo manager logged the incident in the building record.",
        f"We’re including measurements of the affected area and are ready to allow an inspection whenever convenient."
    ]
    return sentence_cleanup(s)

def gen_home_fraud():
    # Subtle cues: no weather match, recent coverage increase, handwritten receipts, same contractor, stock-photo-like images, pre-existing issues suggested
    s = [
        f"I'm submitting a claim for smoke and fire staining we noticed last week in the living room of our house in {pick(cities)}.",
        f"We had adjusted our contents coverage a few weeks prior after redecorating, and luckily the new limits should cover the furniture we listed.",
        f"I didn’t call the fire brigade because the situation calmed quickly; we aired out the space and cleaned up before realizing stains remained.",
        f"The photos I attached were taken on a replacement phone after my old one failed, so the pictures might not have the original dates.",
        f"{pick(contractors)}—who helped us with a similar issue last season—looked at it again and provided a quick handwritten quote while on another job.",
        f"Our {pick(relationships)} mentioned smelling something on {pick(weekdays)}, though they weren’t actually inside at the time.",
        f"We don’t have purchase receipts for the vintage items since many were gifts, but the values should be close based on online listings.",
        f"I also noticed some discolored spots that were there before, but I don’t think they’re related to previous humidity problems."
    ]
    return sentence_cleanup(s)

def gen_life_legit():
    s = [
        f"I'm filing a claim related to my spouse’s hospitalization and subsequent passing; this has been difficult to write.",
        f"The diagnosis was confirmed at {pick(clinics)} and we have the discharge summary, treatment notes, and the attending physician’s report attached.",
        f"The timeline is straightforward: initial symptoms in {pick(weekdays)} of last month, admission two days later, and the outcome the following week.",
        f"Our family coordinated with the funeral home and all invoices are included; amounts remain within the policy limits.",
        f"I’ve verified that the beneficiary information on file matches our notarized documents from earlier this year.",
        f"We are ready to provide any additional medical authorizations you need and can be available for a call at your convenience.",
        f"Please let me know if there is a preferred format for the hospital’s itemized bill so we can resend it cleanly."
    ]
    return sentence_cleanup(s)

def gen_life_fraud():
    # Subtle cues: recent beneficiary change, policy reinstatement after lapse, questionable clinic, reluctance for records, conflicting dates
    s = [
        f"I'm submitting a claim for a serious condition that was identified recently; sorry for the delay, I had trouble collecting some papers.",
        f"We changed the beneficiary details a few weeks before the diagnosis because we moved and my signature looks different after a minor wrist injury.",
        f"The initial certificate came from {pick(clinics)}, which I heard had an accreditation issue last year but they told me it's been sorted out.",
        f"The policy was reinstated after a brief lapse due to a bank switch; we paid the missed premium and everything should be active now.",
        f"Two clinics provided letters with overlapping dates because appointments were rescheduled multiple times; I hope that doesn’t cause confusion.",
        f"I prefer not to release full medical records for privacy reasons at this stage, but summaries should be sufficient to verify the claim.",
        f"I can provide additional statements later—my {pick(relationships)} witnessed some of the symptoms although they weren’t present at the appointment."
    ]
    return sentence_cleanup(s)

def gen_description(branch, is_fraud):
    # Decide length: 6–9 sentences
    if branch == "Auto" and is_fraud:
        base = gen_auto_fraud()
    elif branch == "Auto":
        base = gen_auto_legit()
    elif branch == "Home" and is_fraud:
        base = gen_home_fraud()
    elif branch == "Home":
        base = gen_home_legit()
    elif branch == "Life" and is_fraud:
        base = gen_life_fraud()
    else:
        base = gen_life_legit()

    # Expand with general narrative flourishes to increase variability and length
    adders = [
        "I’m including a rough timeline so it’s easier to review, and I can clarify anything that seems out of order if needed.",
        f"If there’s a preferred claim number to reference, please let me know so I can label the files correctly before uploading again.",
        f"I understand an inspection or call might be required; {pick(first_names)} from our side can also speak to what happened.",
        f"We tried to keep everything as documented as possible, though some details might be fuzzy because the situation was stressful.",
        f"I read through the policy last night and I believe this falls under the coverage section I highlighted in the PDF.",
        f"If you need me to resend photos or scans at a different resolution, I can do that once I’m back at a computer."
    ]
    # Randomly choose 2–4 adders to append
    extra = " ".join(random.sample(adders, k=random.randint(2,4)))
    return f"{base} {extra}"

rows = []
for i in range(1, N + 1):
    branch = np.random.choice(branches, p=branch_probs)
    is_fraud = int(np.random.rand() < fraud_rate)
    desc = gen_description(branch, is_fraud)

    rows.append({
        "Client_ID": client_id(i),
        "Description": desc,
        "Fraud_Label": is_fraud,
        "Branch": branch
    })

df = pd.DataFrame(rows, columns=["Client_ID","Description","Fraud_Label","Branch"])

csv_path = "./data/synthetic_claims.csv"
df.to_csv(csv_path, index=False)


# Show a quick sample
csv_path
