import json

metadata_path = r"C:\Users\laura\OneDrive\Documentos\Dataset\metadata_dfdc_49.json"

with open(metadata_path, "r") as f:
    metadata = json.load(f)

real = 0
fake = 0

for v in metadata.values():
    if v["label"] == "REAL":
        real += 1
    else:
        fake += 1

print("REAL:", real)
print("FAKE:", fake)
print("Ratio REAL:", real / (real + fake))
print("Ratio FAKE:", fake / (real + fake))