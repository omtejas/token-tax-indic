# Dataset handling

Use FLORES-200 through the Hugging Face datasets interface after reviewing the dataset card, source, terms, and current package loading instructions. This project does not redistribute the corpus. Keep downloaded/cache files outside Git (the repository ignores data/raw and data/processed).

The loader uses the English, Hindi, Kannada, Tamil, and Telugu FLORES language configurations and refuses to pair rows if their sentence identifiers do not align. Confirm the dataset schema and revision against the dataset card before a publication run; save the exact dataset revision and split in the run metadata.
