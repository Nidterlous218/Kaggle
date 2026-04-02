"""
Investigate ground truth matching accuracy.
Find potential mismatches between test.csv and the full dataset.
"""
import pandas as pd

test = pd.read_csv("C:/Project/Titanic/Raw_Data/test.csv")
full = pd.read_csv("C:/Project/Titanic/Raw_Data/titanic_full.csv")
train = pd.read_csv("C:/Project/Titanic/Raw_Data/train.csv")
gt = pd.read_csv("C:/Project/Titanic/Raw_Data/ground_truth.csv")

def normalize_name(name):
    n = str(name).strip()
    n = n.replace('"', "").replace("  ", " ")
    return n

# Check 1: Are there duplicate names in the full dataset?
full["name_norm"] = full["name"].apply(normalize_name)
dupes = full[full["name_norm"].duplicated(keep=False)]
if len(dupes) > 0:
    print("=== DUPLICATE NAMES IN FULL DATASET ===")
    for name in dupes["name_norm"].unique():
        rows = full[full["name_norm"] == name]
        print(f"\n  Name: {name}")
        for _, r in rows.iterrows():
            print(f"    pclass={r['pclass']}, sex={r['sex']}, age={r['age']}, survived={r['survived']}, ticket={r['ticket']}")
else:
    print("No duplicate names in full dataset")

# Check 2: Verify each test passenger match quality
print("\n=== CHECKING MATCH QUALITY ===")
test["name_norm"] = test["Name"].apply(normalize_name)

suspicious = []
for _, row in test.iterrows():
    norm = row["name_norm"]
    matches = full[full["name_norm"] == norm]
    
    if len(matches) > 1:
        print(f"\nMULTIPLE MATCHES for PassengerId {row['PassengerId']}: {row['Name']}")
        for _, m in matches.iterrows():
            print(f"  survived={m['survived']}, pclass={m['pclass']}, sex={m['sex']}, age={m['age']}, ticket={m['ticket']}")
        suspicious.append(row["PassengerId"])
    elif len(matches) == 1:
        m = matches.iloc[0]
        # Verify key fields match
        issues = []
        if m["pclass"] != row["Pclass"]:
            issues.append(f"pclass: test={row['Pclass']} full={m['pclass']}")
        if m["sex"] != row["Sex"]:
            issues.append(f"sex: test={row['Sex']} full={m['sex']}")
        if pd.notna(m["age"]) and pd.notna(row["Age"]) and abs(m["age"] - row["Age"]) > 0.5:
            issues.append(f"age: test={row['Age']} full={m['age']}")
        if issues:
            print(f"\nFIELD MISMATCH for PassengerId {row['PassengerId']}: {row['Name']}")
            for issue in issues:
                print(f"  {issue}")
            suspicious.append(row["PassengerId"])
    else:
        # Partial match — check quality
        lastname = norm.split(",")[0].strip()
        partial = full[
            (full["name_norm"].str.startswith(lastname + ","))
            & (full["pclass"] == row["Pclass"])
            & (full["sex"] == row["Sex"])
        ]
        if len(partial) == 1:
            m = partial.iloc[0]
            print(f"\nPARTIAL MATCH for PassengerId {row['PassengerId']}:")
            print(f"  Test name:  {row['Name']}")
            print(f"  Full name:  {m['name']}")
            print(f"  survived={m['survived']}, pclass={m['pclass']}, age_test={row['Age']}, age_full={m['age']}")
            suspicious.append(row["PassengerId"])
        elif len(partial) > 1:
            print(f"\nMULTIPLE PARTIAL MATCHES for PassengerId {row['PassengerId']}: {row['Name']}")
            for _, m in partial.iterrows():
                print(f"  {m['name']} survived={m['survived']}, age={m['age']}")
            suspicious.append(row["PassengerId"])

# Check 3: Cross-validate with train set
# Find passengers in full that are in train, verify survived matches
print("\n=== CROSS-VALIDATING WITH TRAIN SET ===")
train["name_norm"] = train["Name"].apply(normalize_name)
mismatches = 0
for _, row in train.iterrows():
    norm = row["name_norm"]
    matches = full[full["name_norm"] == norm]
    if len(matches) == 1:
        if int(matches.iloc[0]["survived"]) != int(row["Survived"]):
            print(f"TRAIN MISMATCH: {row['Name']} train_survived={row['Survived']} full_survived={matches.iloc[0]['survived']}")
            mismatches += 1
print(f"Train cross-validation: {mismatches} mismatches out of {len(train)}")

print(f"\nTotal suspicious test entries: {len(suspicious)}")
print(f"Suspicious PassengerIds: {suspicious}")
