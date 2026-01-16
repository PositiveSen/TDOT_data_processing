import re
import pandas as pd
import pdfplumber

def extract_october2015_contracts(pdf_path):
    """
    Extract contract information from Const_October2015_Notice.pdf
    Returns list of dictionaries with contract data
    """
    results = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Concatenate all pages
            full_text = ''
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + '\n'
            
            # Pattern to find contracts
            # Format: "COUNTY NAME COUNTY (Contract No.CNP###) Call No. ###"
            # or "COUNTY1 AND COUNTY2 COUNTIES (Contract No.CNP###) Call No. ###"
            
            # Split by contract pattern to get individual contracts
            contract_pattern = r'([A-Z]+(?:\s+AND\s+[A-Z]+)?)\s+(?:COUNTY|COUNTIES)\s+\(Contract\s+No\.([A-Z]{2,3}\d{3,4})\)\s+Call\s+No\.\s+\d+'
            matches = list(re.finditer(contract_pattern, full_text))
            
            for i, match in enumerate(matches):
                contract_num = match.group(2)
                counties = match.group(1).strip()
                
                # Get text section for this contract (from match to next match or end)
                start_pos = match.end()
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(full_text)
                
                section_text = full_text[start_pos:end_pos]
                
                # Extract project description (text before "Project Length")
                desc_match = re.search(r'Project No\.[^\n]+\s+(.+?)\s+Project Length', section_text, re.DOTALL | re.IGNORECASE)
                description = None
                if desc_match:
                    description = desc_match.group(1).strip()
                    # Clean up description
                    description = ' '.join(description.split())
                    description = description.replace('\n', ' ')
                
                # Extract project length
                length_match = re.search(r'Project Length\s*-\s*([\d.]+)\s*(mile[s]?)', section_text, re.IGNORECASE)
                project_length = None
                project_length_unit = None
                if length_match:
                    project_length = length_match.group(1)
                    project_length_unit = 'MILES'
                
                result = {
                    'contract_number': contract_num,
                    'county': counties,
                    'project_description': description,
                    'project_length': project_length,
                    'project_length_unit': project_length_unit
                }
                
                results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []


if __name__ == "__main__":
    pdf_path = 'Data/Const_October2015_Notice.pdf'
    output_file = 'output/missing_data.csv'
    
    print(f"Processing {pdf_path}...")
    contracts = extract_october2015_contracts(pdf_path)
    
    print(f"Found {len(contracts)} contracts\n")
    
    if contracts:
        # Create DataFrame
        df = pd.DataFrame(contracts)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}\n")
        
        # Display results
        print("Extracted contracts:")
        for contract in contracts:
            print(f"  {contract['contract_number']}: {contract['county']}")
            if contract['project_description']:
                print(f"    {contract['project_description'][:80]}...")
            print(f"    Length: {contract['project_length']} {contract['project_length_unit']}")
            print()
    else:
        print("No contracts found")
