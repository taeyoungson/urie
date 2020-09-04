import csv

from pathlib import Path


def read_csv(label_path):
    print(f"loading form {label_path}")
    with open(label_path) as csv_file:
        csv_file_rows = csv.reader(csv_file, delimiter=",")
        for row in csv_file_rows:
            yield row

def main():
    csv_path = Path("./tng_tsfrm_validation.csv.bak")
    crp = "fog"
    with open(csv_path.resolve().as_posix(), "r") as f:
        with open("./tng_tsfrm_validation.csv", "w") as g:
            lines = f.readlines()
            for l in lines:
                splits = l.split(',')

                path = splits[0]
                paths = path.split("/")
                paths[3] = crp
                paths[4] = "3"
                splits[0] = "/".join(paths)

                adjust = ",".join(splits)
                g.write(adjust)
    g.close()

if __name__ == '__main__':
    main()
