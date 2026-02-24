import os
import csv
import json

CONFIG_FILE = "gestures_config.json"
DATASETS_DIR = "datasets"
MONOLITH_CSV = "gesture_data.csv"

def split_csv():
    # Make sure we read the config to know the names
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        classes = config.get("gestures", [])

    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    file_handlers = {}
    csv_writers = {}
    
    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label", "class_name"]

    try:
        with open(MONOLITH_CSV, "r", newline="") as infile:
            reader = csv.reader(infile)
            first_row = next(reader, None)
            
            if not first_row:
                print("CSV is empty.")
                return

            has_class_name = False
            try:
                float(first_row[-1])
            except ValueError:
                if "class_name" in first_row or "label" in first_row:
                    pass
                else:
                    has_class_name = True

            infile.seek(0)
            if first_row and "x" in first_row[0].lower():
                next(reader) 

            for row in reader:
                if not row or len(row) < 64:
                    continue
                
                try:
                    label = int(float(row[63]))
                except ValueError:
                    print(f"Skipping row with invalid label.")
                    continue

                if 0 <= label < len(classes):
                    class_name = classes[label]
                    safe_name = class_name.replace(" ", "_").replace("/", "_")
                    
                    if label not in file_handlers:
                        out_path = os.path.join(DATASETS_DIR, f"{label}_{safe_name}.csv")
                        file_handlers[label] = open(out_path, "w", newline="")
                        csv_writers[label] = csv.writer(file_handlers[label])
                        csv_writers[label].writerow(header)

                    out_row = list(row[:64]) + [class_name]
                    csv_writers[label].writerow(out_row)
                    
        print(f"Successfully split {MONOLITH_CSV} into {DATASETS_DIR}/")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        for f in file_handlers.values():
            f.close()

if __name__ == "__main__":
    split_csv()
