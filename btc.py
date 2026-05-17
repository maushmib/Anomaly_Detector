from pymavlink import DFReader
import csv

log_file = "b1.bin"
output_csv = "flight_log1.csv"

# Open DataFlash log
log = DFReader.DFReader_binary(log_file)

rows = []
index = 0

while True:
    msg = log.recv_msg()

    if msg is None:
        break

    try:
        msg_type = msg.get_type()
        data = msg.to_dict()

        # Put TimeUS before message type
        timeus = data.get("TimeUS", "")

        row = [index, timeus, msg_type]

        # Add remaining fields except TimeUS
        for key, value in data.items():

            if key != "TimeUS":
                row.append(value)

        rows.append(row)
        index += 1

    except:
        pass

# Equalize row lengths
max_len = max(len(r) for r in rows)

for r in rows:
    while len(r) < max_len:
        r.append("")

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Proper DataFlash BIN -> CSV conversion completed!")