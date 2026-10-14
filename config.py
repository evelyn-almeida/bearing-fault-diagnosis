from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
FS = 12_000            # Hz, drive-end accelerometer
WINDOW = 2048          # samples per segment (~0.17 s, about 5 shaft revolutions)
ENVELOPE_BAND = (2_000, 5_000)  # Hz, band around the structural resonances excited by impacts

# SKF 6205-2RS JEM drive-end bearing: fault frequencies as multiples of shaft speed
BEARING = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3570, "FTF": 0.3983}
CLASSES = ["Normal", "Ball", "Inner race", "Outer race"]
