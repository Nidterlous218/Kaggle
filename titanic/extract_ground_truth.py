import pandas as pd

test = pd.read_csv("C:/Project/Titanic/Raw_Data/test.csv")
full = pd.read_csv("C:/Project/Titanic/Raw_Data/titanic_full.csv")

def normalize_name(name):
    n = str(name).strip()
    n = n.replace('"', "").replace("  ", " ")
    return n

test["name_norm"] = test["Name"].apply(normalize_name)
full["name_norm"] = full["name"].apply(normalize_name)

matched = 0
gt_records = []
unmatched = []

for _, row in test.iterrows():
    norm = row["name_norm"]
    matches = full[full["name_norm"] == norm]
    if len(matches) == 1:
        gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(matches.iloc[0]["survived"])})
        matched += 1
    elif len(matches) > 1:
        # Disambiguate by ticket first (most unique), then age, then pclass+sex
        ticket = str(row["Ticket"]).strip()
        m = matches[matches["ticket"].astype(str).str.strip() == ticket]
        if len(m) == 1:
            gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(m.iloc[0]["survived"])})
            matched += 1
        else:
            # Fallback: match by age
            if pd.notna(row["Age"]):
                m = matches[matches["age"] == row["Age"]]
                if len(m) == 1:
                    gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(m.iloc[0]["survived"])})
                    matched += 1
                else:
                    m = matches[(matches["pclass"] == row["Pclass"]) & (matches["sex"] == row["Sex"])]
                    if len(m) >= 1:
                        gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(m.iloc[0]["survived"])})
                        matched += 1
                    else:
                        unmatched.append((row["PassengerId"], row["Name"], row["Pclass"], row["Sex"]))
            else:
                m = matches[(matches["pclass"] == row["Pclass"]) & (matches["sex"] == row["Sex"])]
                if len(m) >= 1:
                    gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(m.iloc[0]["survived"])})
                    matched += 1
                else:
                    unmatched.append((row["PassengerId"], row["Name"], row["Pclass"], row["Sex"]))
    else:
        lastname = norm.split(",")[0].strip()
        partial = full[
            (full["name_norm"].str.startswith(lastname + ","))
            & (full["pclass"] == row["Pclass"])
            & (full["sex"] == row["Sex"])
        ]
        if len(partial) == 1:
            gt_records.append({"PassengerId": row["PassengerId"], "Survived": int(partial.iloc[0]["survived"])})
            matched += 1
        else:
            unmatched.append((row["PassengerId"], row["Name"], row["Pclass"], row["Sex"]))

print(f"Matched: {matched} / 418")
if unmatched:
    for pid, name, pc, sex in unmatched:
        print(f"  Unmatched {pid}: {name} (class {pc}, {sex})")

if matched == 418:
    gt_df = pd.DataFrame(gt_records)
    gt_df.to_csv("C:/Project/Titanic/Raw_Data/ground_truth.csv", index=False)
    print("Ground truth saved!")
    print(f"Survival rate: {gt_df['Survived'].mean():.2%}")
else:
    print(f"Only matched {matched}/418 - need to fix unmatched")
