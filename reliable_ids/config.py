import os

# ===========================
#   CONFIGURATION OPTIONS
# ===========================

# Directory containing the raw dataset files/folders listed in DATASET_INFO below.
# Override with the IDS_DATA_DIR environment variable, e.g.:
#   export IDS_DATA_DIR=/path/to/datasets
DATA_DIR = os.environ.get("IDS_DATA_DIR", "./data")

# Which dataset to run, selected via the DATASET_B environment variable.
DATASET = os.environ.get("DATASET_B")

# ===========================
#        MASTER CONFIG
# ===========================

DATASET_INFO = {
    'nf_ton_iot': {
        'path': os.path.join(DATA_DIR, "NF-ToN-IoT-v3.csv"),
        'attacks': ['ddos', 'dos', 'xss', 'password', 'scanning', 'injection', 'mitm', 'Backdoor', 'ransomware'],
        'nf': True
    },
    'ciciot_23': {
        'path': os.path.join(DATA_DIR, "ciciot_23_dataset") + os.sep,
        'attacks': ['DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce'],
        'nf': False
    },
    'cicddos_2019': {
        'path': os.path.join(DATA_DIR, "cicddos") + os.sep,
        'attacks': ['UDP', 'MSSQL', 'LDAP', 'NetBIOS', 'UDPLag', 'Syn', 'DNS', 'TFTP', 'Portmap', 'NTP', 'SNMP', 'WebDDoS'],
        'nf': False
    }
}


# ===========================
#     AUTO-SELECT SETTINGS
# ===========================

if DATASET not in DATASET_INFO:
    raise ValueError(
        f"DATASET_B={DATASET!r} is not a valid dataset. "
        f"Set the DATASET_B environment variable to one of: {sorted(DATASET_INFO)}"
    )

DATASET_PATH = DATASET_INFO[DATASET]['path']
ALL_ATTACKS = DATASET_INFO[DATASET]['attacks']
NF_DATASET = DATASET_INFO[DATASET]['nf']

# ===========================
#     CONFIRMATION OUTPUT
# ===========================

print(f"Using dataset: {DATASET}")
print(f" -> Path: {DATASET_PATH}")
print(f" -> NF format: {NF_DATASET}")
print(f" -> Attacks: {ALL_ATTACKS}")
