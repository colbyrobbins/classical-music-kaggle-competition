# Data

The competition CSVs are provided by the course and are not committed (see `.gitignore`). Place these files in `data/raw/`:

| File | Contents |
|---|---|
| `train.csv` | account.id + 0/1 label (bought a 2014-15 subscription) |
| `test.csv` | accounts to predict |
| `sample_submission.csv` | submission format (`ID`, `Predicted`) |
| `account.csv` | patron location info and donation history (latin-1 encoded) |
| `subscriptions.csv` | past subscriptions by account |
| `tickets_all.csv` | past tickets by account |
| `concerts.csv` | past concerts by season (UTF-8 with BOM) |
| `concerts_2014-15.csv` | planned 2014-15 sets |
| `zipcodes.csv` | zip code information |
