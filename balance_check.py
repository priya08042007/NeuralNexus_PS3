import os
from collections import defaultdict
print("Hello")
# 👉 Change if needed
labels_path = r"C:\Users\Asus\Projects\Synapse\ai_hackathon\final_dataset\train\labels"

print("\n🔍 DEBUGGING STARTED...\n")

# ✅ Check if path exists
if not os.path.exists(labels_path):
    print("❌ ERROR: Labels path does NOT exist!")
    exit()

print("✅ Path exists")

# ✅ List files
files = os.listdir(labels_path)
print(f"\n📁 Total files in folder: {len(files)}")

if len(files) == 0:
    print("❌ ERROR: Folder is EMPTY!")
    exit()

print("\n📄 Sample files:")
for f in files[:10]:
    print("  ", f)

# Counters
class_counts = defaultdict(int)
total_txt_files = 0
empty_files = 0
non_txt_files = 0

print("\n🔍 Checking each file...\n")

for file in files:
    file_path = os.path.join(labels_path, file)

    # Check extension
    if not file.endswith(".txt"):
        non_txt_files += 1
        continue

    total_txt_files += 1
    print(f"📄 Reading: {file}")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

            if len(lines) == 0:
                print(f"⚠️ Empty file: {file}")
                empty_files += 1
                continue

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5:
                    print(f"⚠️ Invalid format in {file}: {line}")
                    continue

                class_id = parts[0]
                class_counts[class_id] += 1

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# 📊 Results
print("\n📊 FINAL RESULTS")
print("--------------------------------------------------")
print(f"Total .txt label files: {total_txt_files}")
print(f"Non-txt files (ignored): {non_txt_files}")
print(f"Empty label files: {empty_files}")

# Class distribution
if class_counts:
    print("\n📊 Class Distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"Class {cls}: {count} objects")

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    print("\n⚖️ Balance Analysis:")
    print(f"Max class count: {max_count}")
    print(f"Min class count: {min_count}")

    if max_count - min_count < 0.2 * max_count:
        print("✅ Dataset is fairly BALANCED")
    else:
        print("⚠️ Dataset is IMBALANCED")

else:
    print("\n❌ No valid class data found!")