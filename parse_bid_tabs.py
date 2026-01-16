import pdfplumber
import pypdf
import re
import os
import warnings
import io

# Suppress PDF decompression warnings
warnings.filterwarnings('ignore', message='.*Data-loss while decompressing.*')


def extract_bid_info(pdf_path):
    """
    Extract all contract numbers, project descriptions, and project lengths from a bid tab PDF.
    Optimized to only read project info pages, skipping item data pages.
    
    Parameters:
    -----------
    pdf_path : str
        Path to the PDF file
        
    Returns:
    --------
    list
        List of dictionaries, each containing contract_number, project_description, and project_length
    """
    results = []
    
    try:
        # First, try with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Only check pages that likely contain contract info (pages with "COUNTY" keyword)
            # Skip pages with only item data (they have "UNIT PRICE" but no "COUNTY")
            contract_pages = []
            
            for i, page in enumerate(pdf.pages):
                # Extract just first 500 chars to check if it's a contract info page
                try:
                    text_sample = page.extract_text()
                    if not text_sample:
                        # If pdfplumber fails, try pypdf as fallback
                        with open(pdf_path, 'rb') as f:
                            pypdf_reader = pypdf.PdfReader(f)
                            text_sample = pypdf_reader.pages[i].extract_text()
                    
                    text_sample = text_sample[:500] if text_sample else ""
                    # Check for either COUNTY or COUNTIES and Contract No. (handle line breaks)
                    if ('COUNTY' in text_sample or 'COUNTIES' in text_sample) and re.search(r'Contract\s+No\.', text_sample):
                        contract_pages.append(i)
                except Exception as e:
                    print(f"  Warning: Error reading page {i}: {str(e)}")
                    continue
            
            # Extract full text only from contract info pages
            for page_num in contract_pages:
                try:
                    page_text = pdf.pages[page_num].extract_text()
                    
                    # If pdfplumber extraction is empty or very short, try pypdf
                    if not page_text or len(page_text) < 100:
                        with open(pdf_path, 'rb') as f:
                            pypdf_reader = pypdf.PdfReader(f)
                            page_text = pypdf_reader.pages[page_num].extract_text()
                except Exception as e:
                    print(f"  Warning: Error extracting text from page {page_num}: {str(e)}")
                    continue
                
                # Skip if page text is still empty or too short
                if not page_text or len(page_text) < 50:
                    print(f"  Warning: Page {page_num} has insufficient text")
                    continue
                
                # Find ALL contract info on this page (there may be multiple contracts per page)
                # Handle both "COUNTY" (single) and "COUNTIES" (plural) patterns
                # Pattern 1: Single county - "COUNTY_NAME COUNTY (Contract No. XXX###"
                # Pattern 2: Multiple counties - "COUNTIES NAME1, NAME2, ... (Contract No. XXX###"  
                # Pattern 3: Multiple counties reversed - "NAME1, NAME2, ... COUNTIES (Contract No. XXX###"
                # Pattern 4: Multiple counties with AND - "NAME1 AND NAME2 COUNTIES (Contract No. XXX###"
                contract_matches = []
                
                # Try pattern for "COUNTY_NAMES COUNTIES (Contract No. ...)" - handles both comma and AND separated
                matches_reverse = list(re.finditer(r'([A-Z\s,AND]+?)\s+COUNTIES\s+\(Contract\s+No\.\s+([A-Z]{2,3}\d{3,4})(?:\s+Call\s+\d+)?\)', page_text, re.IGNORECASE | re.DOTALL))
                for m in matches_reverse:
                    contract_matches.append(m)
                
                # Try pattern for "COUNTIES COUNTY_NAMES (Contract No. ...)"
                matches_plural = list(re.finditer(r'COUNTIES\s+([A-Z\s,AND]+?)\(Contract\s+No\.\s+([A-Z]{2,3}\d{3,4})(?:\s+Call\s+\d+)?\)', page_text, re.IGNORECASE | re.DOTALL))
                for m in matches_plural:
                    contract_matches.append(m)
                
                # Try pattern for singular COUNTY
                matches_singular = list(re.finditer(r'([A-Z]+)\s+COUNTY\s+\(Contract\s+No\.\s+([A-Z]{2,3}\d{3,4})(?:\s+Call\s+\d+)?\)', page_text, re.IGNORECASE | re.DOTALL))
                for m in matches_singular:
                    contract_matches.append(m)
                
                for contract_match in contract_matches:
                    # Extract contract number and county - different groups depending on pattern
                    counties = contract_match.group(1).strip()
                    contract_num = contract_match.group(2)
                    
                    result = {
                        'contract_number': contract_num,
                        'county': counties,
                        'project_description': None,
                        'project_length': None,
                        'project_length_unit': None,
                        'file_name': os.path.basename(pdf_path)
                    }
                    
                    # Get text after this contract match for extraction
                    # Look for the section between this contract and the next one (or end of page)
                    start_pos = contract_match.end()
                    # Find next contract or end of text (look for any pattern)
                    next_match = re.search(r'(?:[A-Z\s,AND]+?\s+COUNTIES|COUNTIES\s+[A-Z\s,AND]+?|(?:\w+)\s+COUNTY)\s+\(Contract No\.', page_text[start_pos:], re.IGNORECASE)
                    if next_match:
                        section_text = page_text[start_pos:start_pos + next_match.start()]
                    else:
                        section_text = page_text[start_pos:]
                    
                    # Extract project description (with or without "THE")
                    # Skip project codes that appear before the description
                    desc_match = re.search(r'(?:THE\s+)?(.+?)\s+PROJECT\s+LENGTH', section_text, re.IGNORECASE | re.DOTALL)
                    if desc_match:
                        description = desc_match.group(1)
                        description = ' '.join(description.split())
                        
                        # Remove everything up to and including lines of comma-separated project codes
                        # Handles: ") NH-I-75-1(155),33I075-F1-006,33I075-F2-006,33I075-F3-004 I-75 INTERCHANGE..."
                        # Pattern matches closing paren followed by codes with commas, then captures the actual description
                        code_line_match = re.match(r'^.*?\)\s*(?:[A-Z0-9\-\(\),/]+,\s*)+[A-Z0-9\-\(\),/]+\s+(.+)', description)
                        if code_line_match:
                            description = code_line_match.group(1)
                        
                        # Remove everything before "THE" only if preceded by project codes/n/a
                        # This handles patterns like: "62S068-M8-004,n/a,62S307-M8-002,STP/HSIP-2(273) THE RESURFACING..."
                        # But NOT: "RESURFACING ON I-269 FROM THE FAYETTE COUNTY LINE" (THE after real work description)
                        # And NOT: "MICROSURFACING ON U.S. 11 (U.S. 64, S.R. 2) FROM THE..." (THE after route with commas)
                        # Only remove if prefix contains project code pattern (alphanumeric with hyphens/slashes followed by comma)
                        # Or contains "n/a"
                        # Use count=1 to only remove before the FIRST "THE", not all occurrences
                        the_match = re.search(r'^(.+?)THE\s+', description)
                        if the_match:
                            prefix = the_match.group(1)
                            # Check for project code patterns: codes like "STP-123," or "n/a,"
                            # Must have comma followed by more content (not just route designation commas)
                            if re.search(r'([A-Z0-9]{3,}[\-/][A-Z0-9\-/\(\)]+,|n/a,)', prefix, re.IGNORECASE):
                                description = re.sub(r'^.*?(?=THE\s+)', '', description, count=1)
                        
                        # If description doesn't start with "THE", remove other code patterns
                        if not description.startswith('THE'):
                            # Remove leading commas and whitespace
                            description = re.sub(r'^[,\s]+', '', description)
                            
                            # Remove "n/a," prefix if it exists
                            description = re.sub(r'^n/a,\s*', '', description, flags=re.IGNORECASE)
                            
                            # Remove project codes with commas - patterns like "62S068-M8-004," or "STP/HSIP-2(273),"
                            # Fixed: removed \s from character class to avoid matching "I-75 INTERCHANGE MODIFICATION AT I-24,"
                            description = re.sub(r'^(?:[A-Z0-9]+[\-/\(][A-Z0-9\-/\(\),]*?,\s*)+', '', description)
                            
                            # Remove standalone project codes followed by space and capital words
                            # Handles: "73008-3243-14 RECONSTRUCTION" or "R3IVAR-M3-007 PERFORMANCE"
                            # But NOT highway designations like "I-75" or "SR-1" (requires 4+ chars before hyphen)
                            description = re.sub(r'^[A-Z0-9]{4,}[\-/][A-Z0-9\-/]+\s+(?=[A-Z]{2,})', '', description)
                        
                        result['project_description'] = description.strip()
                    
                    # Extract project length (handles both MILES and KILOMETERS)
                    length_match = re.search(r'PROJECT\s+LENGTH\s*-\s*(\d+\.?\d*)\s*(MILES?|KILOMETERS?)', section_text, re.IGNORECASE)
                    if length_match:
                        result['project_length'] = length_match.group(1)
                        result['project_length_unit'] = length_match.group(2).upper()
                    
                    # Skip contracts with "No Valid Bids" or "PROPOSAL REJECTED"
                    if re.search(r'(No Valid Bids|PROPOSAL REJECTED)', section_text, re.IGNORECASE):
                        continue
                    
                    results.append(result)
        
        # If no results found, return empty result with filename for debugging
        if not results:
            return [{'contract_number': None, 'county': None, 'project_description': None, 'project_length': None, 'project_length_unit': None, 'file_name': os.path.basename(pdf_path)}]
        
        return results
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return [{'contract_number': None, 'county': None, 'project_description': None, 'project_length': None, 'file_name': os.path.basename(pdf_path), 'error': str(e)}]


if __name__ == "__main__":
    # Test with the first PDF file
    test_pdf = "Data/Bid Tabs/2023/20230602_SummaryOfBids.pdf"
    
    if os.path.exists(test_pdf):
        print(f"\nTesting with: {test_pdf}\n")
        info = extract_bid_info(test_pdf)
        
        print("\n" + "=" * 80)
        print("EXTRACTED INFORMATION:")
        print("=" * 80)
        print(f"File Name: {info['file_name']}")
        print(f"Contract Number: {info['contract_number']}")
        print(f"Project Description: {info['project_description']}")
        print(f"Project Length: {info['project_length']}")
        if 'error' in info:
            print(f"Error: {info['error']}")
    else:
        print(f"File not found: {test_pdf}")
        print("\nLooking for available PDF files...")
        # Find first available PDF
        for root, dirs, files in os.walk("Data/Bid Tabs"):
            for file in files:
                if file.endswith('.pdf'):
                    test_file = os.path.join(root, file)
                    print(f"\nFound: {test_file}")
                    info = extract_bid_info(test_file)
                    print("\n" + "=" * 80)
                    print("EXTRACTED INFORMATION:")
                    print("=" * 80)
                    print(f"File Name: {info['file_name']}")
                    print(f"Contract Number: {info['contract_number']}")
                    print(f"Project Description: {info['project_description']}")
                    print(f"Project Length: {info['project_length']}")
                    break
            break
