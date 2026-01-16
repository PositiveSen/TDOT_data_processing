import pandas as pd
import os


def find_unique_proposals(csv_file='Data/TDOT_data.csv', output_file='output/unique_proposals.txt', print_results=True):
    """
    Find and return unique proposals from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file to analyze
    output_file : str
        Path to save the results (optional, set to None to skip saving)
    print_results : bool
        Whether to print the results to console
        
    Returns:
    --------
    list or None
        Sorted list of unique proposals, or None if no proposal column found
    """
    # Read the CSV file with encoding handling
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
        except:
            df = pd.read_csv(csv_file, encoding='cp1252')
    
    if print_results:
        print("Column names:")
        print(df.columns.tolist())
        print("\n")
    
    # Check if there's a 'Proposal' column (case-insensitive search)
    proposal_col = None
    for col in df.columns:
        if 'proposal' in col.lower():
            proposal_col = col
            break
    
    if proposal_col:
        # Find letting date column
        letting_date_col = None
        for col in df.columns:
            if 'letting' in col.lower() and 'date' in col.lower():
                letting_date_col = col
                break
        
        # Get unique proposals with letting dates
        if letting_date_col:
            unique_proposals_df = df[[proposal_col, letting_date_col]].drop_duplicates()
            unique_proposals_df = unique_proposals_df[unique_proposals_df[proposal_col].notna()]
            unique_proposals_df = unique_proposals_df.sort_values(proposal_col)
        else:
            # Fallback to just unique proposals if no letting date found
            unique_proposals = df[proposal_col].unique()
            unique_proposals_sorted = sorted([str(p) for p in unique_proposals if pd.notna(p)])
            unique_proposals_df = pd.DataFrame({proposal_col: unique_proposals_sorted})
        
        if print_results:
            print(f"Found {len(unique_proposals_df)} unique proposals in column '{proposal_col}':")
            if letting_date_col:
                print(f"Including letting dates from column '{letting_date_col}'")
            print("\n")
            for i, (idx, row) in enumerate(unique_proposals_df.iterrows(), 1):
                if letting_date_col:
                    print(f"{i}. {row[proposal_col]} - {row[letting_date_col]}")
                else:
                    print(f"{i}. {row[proposal_col]}")
        
        # Save to a file if output_file is specified
        if output_file:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w') as f:
                f.write(f"Unique Proposals from {proposal_col}\n")
                f.write("=" * 50 + "\n\n")
                for i, (idx, row) in enumerate(unique_proposals_df.iterrows(), 1):
                    if letting_date_col:
                        f.write(f"{i}. {row[proposal_col]} - {row[letting_date_col]}\n")
                    else:
                        f.write(f"{i}. {row[proposal_col]}\n")
            
            if print_results:
                print(f"\n\nUnique proposals saved to '{output_file}'")
        
        return unique_proposals_df
    else:
        if print_results:
            print("No column with 'proposal' in the name was found.")
            print("Please check the column names above and modify the script accordingly.")
        return None


if __name__ == "__main__":
    find_unique_proposals()
