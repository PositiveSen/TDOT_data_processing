# Data Directory

This directory should contain the TDOT data files:

## Required Files:
- `TDOT_data.csv` - Main dataset (excluded from repo due to size ~115MB)
- `Pavement Marking.csv` - Subset data (excluded from repo) 
- `Bid Tabs/` - PDF files by year (excluded from repo)

## Getting Started:
1. Obtain the TDOT_data.csv file and place it here
2. Run the data processing scripts to populate this directory
3. The categorization scripts will generate additional CSV files here

## File Structure:
```
Data/
├── TDOT_data.csv          # Main dataset (not in repo)
├── Pavement Marking.csv   # Subset data (not in repo)  
└── Bid Tabs/             # PDF files by year (not in repo)
    ├── 2014/
    ├── 2015/
    ├── ...
    └── 2025/
```

Note: Large data files are excluded from the repository but remain essential for running the analysis.