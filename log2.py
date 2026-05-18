import csv

input_file = "output3.csv"
output_file = "gps_trajectory.csv"

with open(input_file, "r") as f, open(output_file, "w", newline="") as out:

    reader = csv.reader(f)
    writer = csv.writer(out)

    writer.writerow(["time", "lat", "lon", "alt", "speed"])

    for row in reader:

        try:

            if row[2] == "GPS":

                time = row[1]

                lat = float(row[10])
                lon = float(row[11])
                alt = float(row[12])

                speed = float(row[9])

                writer.writerow([
                    time,
                    lat,
                    lon,
                    alt,
                    speed
                ])

        except:
            pass

print("Correct GPS trajectory extracted")