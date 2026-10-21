# Data Directory

Download the LendingClub `accepted_2007_to_2018Q4.csv.gz` ship file from Kaggle,
decompress it, and rename it to `data/raw/loan.csv` — that is the path
`src/config.py::DATA_RAW` points to and what `make data` loads via
`psycopg3 COPY`.

### Kaggle CLI Command
```bash
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/raw/
gunzip data/raw/accepted_2007_to_2018Q4.csv.gz
mv data/raw/accepted_2007_to_2018Q4.csv data/raw/loan.csv
```
