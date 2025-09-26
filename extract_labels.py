import os

# path to your merged dataset
base_dir = r"C:/Users/devap/OneDrive/Desktop/MajorProject/intelligent_road_safety/datasets/merged"

# folders to check
splits = ["train/labels", "val/labels", "test/labels"]

for split in splits:
    label_dir = os.path.join(base_dir, split)
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls = int(parts[0])
                    # remap pothole (originally 0) → 8
                    if "pothole" in file.lower() or "potholes" in file.lower():
                        cls = 8
                    new_lines.append(" ".join([str(cls)] + parts[1:]) + "\n")
            
            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("✅ All pothole labels remapped to class 8")
