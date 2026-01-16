import pdfplumber
import re

pdf_path = "/home/senzhang/Projects/TDOT_ALL_DATA_PROCESS/Data/Bid Tabs/2022/DB2101_SummaryOfBids.pdf"
with pdfplumber.open(pdf_path) as pdf:
    page_text = pdf.pages[0].extract_text()
    
    print("=== Testing singular COUNTY pattern ===")
    # Try pattern for singular COUNTY
    matches_singular = list(re.finditer(r'([A-Z]+)\s+COUNTY\s+\(Contract\s+No\.\s+([A-Z]{2,3}\d{3,4})(?:\s+Call\s+\d+)?\)', page_text, re.IGNORECASE | re.DOTALL))
    print(f"Found {len(matches_singular)} matches with singular pattern")
    
    if matches_singular:
        contract_match = matches_singular[0]
        print("Contract match found:")
        print(f"  County: {contract_match.group(1).strip()}")
        print(f"  Contract: {contract_match.group(2)}")
        print(f"  Match ends at position: {contract_match.end()}")
        
        # Get section text after match
        start_pos = contract_match.end()
        section_text = page_text[start_pos:start_pos + 300]
        print("\nSection text after match (first 300 chars):")
        print(repr(section_text))
        
        # Extract description
        desc_match = re.search(r'(?:THE\s+)?(.+?)\s+PROJECT\s+LENGTH', section_text, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1)
            print("\nRaw extracted description:")
            print(repr(description))
            
            # Clean whitespace
            description = ' '.join(description.split())
            print("\nAfter whitespace cleanup:")
            print(repr(description))
            
            # Try code line removal
            code_line_match = re.match(r'^.*?\)\s*(?:[A-Z0-9\-\(\),/]+,\s*)+[A-Z0-9\-\(\),/]+\s+(.+)', description)
            if code_line_match:
                print("\nCode line pattern MATCHED!")
                description = code_line_match.group(1)
                print("After code line removal:")
                print(repr(description))
            else:
                print("\nCode line pattern DID NOT MATCH")

        print("Contract match found:")
        print(f"  County: {contract_match.group(1).strip()}")
        print(f"  Contract: {contract_match.group(2)}")
        print(f"  Match ends at position: {contract_match.end()}")
        
        # Get section text after match
        start_pos = contract_match.end()
        section_text = page_text[start_pos:start_pos + 500]
        print("\nSection text (first 500 chars):")
        print(repr(section_text))
        
        # Extract description
        desc_match = re.search(r'(?:THE\s+)?(.+?)\s+PROJECT\s+LENGTH', section_text, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1)
            print("\nRaw extracted description:")
            print(repr(description))
            
            # Clean whitespace
            description = ' '.join(description.split())
            print("\nAfter whitespace cleanup:")
            print(repr(description))
            
            # Try code line removal
            code_line_match = re.match(r'^.*?\)\s*(?:[A-Z0-9\-\(\),/]+,\s*)+[A-Z0-9\-\(\),/]+\s+(.+)', description)
            if code_line_match:
                print("\nCode line pattern MATCHED!")
                description = code_line_match.group(1)
                print("After code line removal:")
                print(repr(description))
            else:
                print("\nCode line pattern DID NOT MATCH")
                print("\nLet's examine the start of the description more closely:")
                print(description[:100])
