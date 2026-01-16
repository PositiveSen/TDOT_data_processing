import pandas as pd
import re

# Configuration
input_file = 'output/bid_tabs_combined.csv'
output_file = 'output/bid_tabs_combined_categorized.csv'

def categorize_work_type(description):
    """
    Categorize project into primary work type based on description.
    """
    if pd.isna(description):
        return 'Other'
    
    desc = str(description).upper()
    
    # Check in order of specificity (most specific first)
    
    # 1. Bridge Work - check first for descriptions that START with bridge construction
    # Match "THE CONSTRUCTION OF A [type] BRIDGE" or "CONSTRUCTION OF [number] BRIDGE" at the beginning
    if (desc.startswith('THE CONSTRUCTION OF A') and 'BRIDGE' in desc[:80] and 'GRADING' not in desc[:80]) or \
       (desc.startswith('CONSTRUCTION OF') and 'BRIDGE' in desc[:60] and 'GRADING' not in desc[:60]):
        return 'Bridge Work'
    
    # 2. Drainage/Culvert Work (drainage structures, culvert repair/replacement, sliplining) - check before Roadway Construction
    if ('DRAINAGE STRUCTURE' in desc or 'DRAINAGE IMPROVEMENT' in desc or 
        'CULVERT REPAIR' in desc or 'CULVERT' in desc and ('REPLACEMENT' in desc or 'SLIPLINING' in desc) or 
        'SLIPLINING OF CULVERT' in desc or 'BOX CULVERT' in desc or 
        ('CORRUGATED METAL PIPE' in desc and ('REPAIR' in desc or 'REPLACEMENT' in desc)) or
        'CMP CULVERT' in desc):
        return 'Drainage/Culvert Work'
    
    # 3. Roadway Construction (grading, reconstruction projects)
    # Removed generic 'CONSTRUCTION OF' + 'BRIDGE' check - now handled above
    if ('GRADING' in desc or 
        'RECONSTRUCTION' in desc or
        ('CONSTRUCTION OF' in desc and ('ROUNDABOUT' in desc or 
                                         'PEDESTRIAN' in desc or 'INTERCHANGE' in desc))):
        return 'Roadway Construction'
    
    # 3. Pavement Preservation (microsurfacing, chip seal, slurry seal, crack seal, patching, joint stabilization, PCC repair, pavement repair, concrete pavement repair, sealing)
    if ('MICROSURFAC' in desc or 'MICRO-SURFAC' in desc or 'CHIP SEAL' in desc or 'SLURRY' in desc or 
        'CRACK SEAL' in desc or 'PATCHING' in desc or 'JOINT STABILIZATION' in desc or
        ('PCC' in desc and 'REPAIR' in desc) or
        ('REMOVAL OF PORTLAND CEMENT CONCRETE' in desc and 'REPAVING' in desc) or
        ('REPAIR OF PORTLAND CEMENT CONCRETE PAVEMENT' in desc) or
        ('PAVEMENT' in desc and 'REPAIR' in desc) or 
        'CONCRETE PAVEMENT REPAIR' in desc or 'CONCRETE PAVING REPAIR' in desc or
        ('JOINT' in desc and ('RESEAL' in desc or 'SEALING' in desc)) or
        ('SEALING ON' in desc)):
        return 'Pavement Preservation'
    
    # 4. Safety Improvement (safety improvement projects - check before resurfacing)
    if ('SAFETY' in desc and 'IMPROVEMENT' in desc):
        return 'Safety Improvement'
    
    # 5. Resurfacing (check before bridge work to catch resurfacing projects with incidental bridge work)
    # Projects starting with "THE RESURFACING" are primarily resurfacing even if they include bridge repair
    # Full depth reclamation is a resurfacing technique
    if ('THE RESURFACING' in desc or 'THE RESURFACE' in desc or
        'FULL DEPTH RECLAMATION' in desc or
        (('RESURFACING' in desc or 'RESURFACE' in desc) and 'INCLUDING BRIDGE' in desc)):
        return 'Resurfacing'
    
    # 6. Roadway Improvements (widening, intersection/interchange improvements, ramp improvements/repairs, turn lanes, ADA curb ramps)
    # Check this BEFORE Bridge Work to catch projects that are primarily roadway work with incidental bridge work
    if ('WIDENING' in desc or
        (('IMPROVEMENT' in desc or 'IMPROVEMENTS' in desc) and 
         ('INTERSECTION' in desc or 'INTERCHANGE' in desc or 'RAMP' in desc or 'GRADE CROSSING' in desc)) or
        ('RAMP' in desc and 'REPAIR' in desc) or
        ('TURN LANE' in desc and 'CONSTRUCTION' in desc) or
        'ADA' in desc or 'CURB RAMP' in desc):
        return 'Roadway Improvements'
    
    # 7. Bridge Work (bridge repair/replacement/rehab as primary work, excluding construction which is handled above)
    if 'BRIDGE' in desc and ('REPAIR' in desc or 'REPLACEMENT' in desc or 'REHAB' in desc):
        return 'Bridge Work'
    
    # 8. Resurfacing - catch remaining resurfacing projects (after bridge work check)
    if (('RESURFACING' in desc or 'RESURFACE' in desc or 'OVERLAY' in desc or 'PAVING' in desc) and 
        'REPAIR' not in desc):
        return 'Resurfacing'
    
    # 9. Slope/Rockfall Stabilization (slope stabilization, rockfall mitigation, slide repair, soil nails)
    if ('SLOPE STABILIZATION' in desc or 'ROCKFALL' in desc or 'SLIDE REPAIR' in desc or 'SLIDE' in desc or
        'SOIL NAIL' in desc or 'ROCK ANCHOR' in desc or 'EMERGENCY SLOPE' in desc or 
        'EMERGENCY SLIDE' in desc or 'ROCKFALL MITIGATION' in desc or 'SLOPE REPAIR' in desc):
        return 'Slope/Rockfall Stabilization'
    
    # 10. Maintenance (mowing, litter removal, sweeping, drain cleaning, tunnel cleaning, fence repair, landscaping, sinkhole)
    if ('MOWING' in desc or 'LITTER' in desc or 'SWEEPING' in desc or 'DRAIN CLEANING' in desc or
        'TUNNEL CLEANING' in desc or ('FENCE' in desc and 'REPAIR' in desc) or
        'LANDSCAPING' in desc or 'SINKHOLE' in desc):
        return 'Maintenance'
    
    # 11. ITS/Electrical (intelligent transportation systems, signals, lighting)
    if ('ITS' in desc or 'I.T.S.' in desc or 'SMARTWAY' in desc or 'INTELLIGENT TRANSPORTATION' in desc or 
        'DYNAMIC MESSAGE SIGN' in desc or 'DMS' in desc or 
        ('LIGHTING' in desc and ('INTERCHANGE' in desc or 'TUNNEL' in desc)) or
        ('SIGNAL' in desc and ('INSTALLATION' in desc or 'UPGRADE' in desc or 'MODERNIZATION' in desc)) or
        'SIGNALIZATION' in desc or 'VEHICLE DETECTION' in desc or 'WRONG WAY PREVENTION' in desc):
        return 'ITS/Electrical'
    
    # 12. Pavement Marking (striping, marking, pavement markers, HOV markings)
    # Removed "On-Call Services" - these should be categorized by their actual work type
    if ((('MARKING' in desc or 'STRIPING' in desc or 'MARKER' in desc or 'RELENSING' in desc) and 'PAVEMENT' in desc) or
        'HOV MARKING' in desc):
        return 'Pavement Marking'
    
    # 13. Safety Hardware (signs, guardrails, barriers, attenuators)
    if ('OVERHEAD SIGN' in desc or 'CANTILEVER SIGN' in desc or 'SIGN STRUCTURE' in desc or 
        'FLASHING BEACON' in desc or 'FLASHING SIGN' in desc or 'SCHOOL SIGN' in desc or 
        'SPEED LIMIT SIGN' in desc or 'CABLE BARRIER' in desc or 'NOISE WALL' in desc or
        'EMERGENCY REFERENCE MARKER' in desc or 'SIGNING IMPROVEMENTS' in desc or
        'GUARDRAIL' in desc or 'ATTENUATOR' in desc or
        ('INSTALLATION OF' in desc and 'SIGNS' in desc) or ('REPLACEMENT OF' in desc and 'SIGNS' in desc) or
        'HOV SIGN' in desc or 'WARNING SYSTEM' in desc or 'SIGNING' in desc):
        return 'Safety Hardware'
    
    # 14. Facility/Yard Infrastructure (salt bins, brine sheds, maintenance facilities, TDOT buildings)
    if ('SALT BIN' in desc or 'BRINE SHED' in desc or 'SALT BRINE SHED' in desc or 
        ('WEIGH STATION' in desc and 'IMPROVEMENT' in desc) or
        ('BUILDING' in desc and ('MAINTENANCE' in desc or 'TDOT' in desc or 'T.D.O.T.' in desc))):
        return 'Facility/Yard Infrastructure'
    
    # Default
    return 'Other'


def extract_component_flags(description):
    """
    Extract binary flags for project components.
    Returns a dictionary of flags.
    """
    if pd.isna(description):
        return {}
    
    desc = str(description).upper()
    
    flags = {
        'has_grading': 'GRADING' in desc,
        'has_drainage': 'DRAINAGE' in desc or 'CULVERT' in desc,
        'has_bridge': 'BRIDGE' in desc,
        'has_retaining_wall': 'RETAINING WALL' in desc,
        'has_paving': 'PAVING' in desc or 'SURFACING' in desc or 'OVERLAY' in desc,
        'has_guardrail': 'GUARDRAIL' in desc,
        'has_signs': 'SIGN' in desc and 'DESIGN' not in desc,
        'has_lighting': 'LIGHTING' in desc,
        'has_signals': 'SIGNAL' in desc,
        'has_pavement_marking': 'MARKING' in desc or 'STRIPING' in desc,
        'has_mowing': 'MOWING' in desc,
        'has_chip_seal': 'CHIP SEAL' in desc,
        'has_crack_seal': 'CRACK SEAL' in desc,
        'is_on_call': 'ON-CALL' in desc or 'ON CALL' in desc,
        'is_emergency': 'EMERGENCY' in desc,
    }
    
    return flags


def extract_route_info(description):
    """
    Extract route type information.
    Returns the primary road type: Interstate, US, State, or Other
    """
    if pd.isna(description):
        return 'Other'
    
    desc = str(description).upper()
    
    # Check for interstate (highest priority)
    if re.search(r'\bI-\d+\b', desc):
        return 'Interstate'
    
    # Check for US route
    if re.search(r'\bU\.S\.\s+\d+\b', desc):
        return 'US'
    
    # Check for state route
    if re.search(r'\bS\.R\.\s+\d+\b', desc):
        return 'State'
    
    return 'Other'


def count_counties(county_str):
    """
    Count number of counties in the project.
    """
    if pd.isna(county_str):
        return 0
    
    # Split by comma and 'AND'
    counties = re.split(r',|\sAND\s', str(county_str))
    # Filter out empty strings and clean
    counties = [c.strip() for c in counties if c.strip()]
    
    return len(counties)


def categorize_all_projects(input_csv, output_csv, unique_proposals_file=None):
    """
    Read bid tabs data, categorize projects, and save to CSV with new columns.
    """
    print("Loading data...")
    df = pd.read_csv(input_csv)
    
    # Filter by unique proposals if provided
    if unique_proposals_file:
        print(f"Loading unique proposals from {unique_proposals_file}...")
        with open(unique_proposals_file, 'r') as f:
            lines = f.readlines()
        
        # Extract contract numbers (skip header lines and parse numbered list)
        unique_contracts = []
        for line in lines:
            line = line.strip()
            # Skip header and empty lines
            if not line or line.startswith('=') or line.startswith('Unique'):
                continue
            # Extract contract number from numbered list format: "1. CNN001" or "1. CNN001 - 20140110"
            parts = line.split('.', 1)
            if len(parts) == 2:
                # Split by ' - ' to handle letting date if present
                contract_part = parts[1].strip().split(' - ')[0]
                unique_contracts.append(contract_part)
        
        print(f"Found {len(unique_contracts)} unique proposals")
        print(f"Filtering from {len(df)} to only unique proposals...")
        df = df[df['contract_number'].isin(unique_contracts)]
        print(f"Filtered to {len(df)} projects")
    
    print(f"Processing {len(df)} projects...")
    
    # Add work type
    print("Categorizing work types...")
    df['work_type'] = df['project_description'].apply(categorize_work_type)
    
    # Add component flags
    print("Extracting component flags...")
    component_flags = df['project_description'].apply(extract_component_flags)
    for flag in ['has_grading', 'has_drainage', 'has_bridge', 'has_retaining_wall', 'has_paving',
                 'has_guardrail', 'has_signs', 'has_lighting', 'has_signals', 
                 'has_pavement_marking', 'has_barrier', 'has_concrete', 'has_mowing',
                 'has_chip_seal', 'has_crack_seal', 'is_on_call', 'is_emergency']:
        df[flag] = component_flags.apply(lambda x: 1 if x.get(flag, False) else 0)
    
    # Add route information
    print("Extracting route information...")
    df['road_type'] = df['project_description'].apply(extract_route_info)
    
    # Add county count
    print("Counting counties...")
    df['num_counties'] = df['county'].apply(count_counties)
    df['multi_county'] = (df['num_counties'] > 1).astype(int)
    
    # Reorder columns
    column_order = [
        'contract_number',
        'county',
        'num_counties',
        'multi_county',
        'work_type',
        'road_type',
        'project_description',
        'project_length',
        'project_length_unit',
        'has_grading',
        'has_drainage',
        'has_bridge',
        'has_retaining_wall',
        'has_paving',
        'has_guardrail',
        'has_signs',
        'has_lighting',
        'has_signals',
        'has_pavement_marking',
        'has_barrier',
        'has_concrete',
        'has_mowing',
        'has_chip_seal',
        'has_crack_seal',
        'is_on_call',
        'is_emergency',
    ]
    
    df = df[column_order]
    
    # Sort by work_type
    df = df.sort_values('work_type')
    
    # Save to CSV
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("CATEGORIZATION SUMMARY")
    print("="*80)
    print(f"\nTotal projects: {len(df)}")
    print("\nWork Type Distribution:")
    print(df['work_type'].value_counts().to_string())
    
    print("\n\nComponent Flags Summary:")
    for flag in ['has_grading', 'has_drainage', 'has_bridge', 'has_retaining_wall', 'has_paving',
                 'has_guardrail', 'has_signs', 'has_lighting', 'has_signals', 
                 'has_pavement_marking', 'has_barrier', 'has_concrete', 'has_mowing',
                 'has_chip_seal', 'has_crack_seal', 'is_on_call', 'is_emergency']:
        count = df[flag].sum()
        pct = count / len(df) * 100
        print(f"  {flag:25} {count:5} ({pct:5.1f}%)")
    
    print("\n\nRoute Type Summary:")
    print(df['road_type'].value_counts().to_string())
    
    print("\n\nCounty Distribution:")
    print(f"  Single county:  {(df['num_counties'] == 1).sum()}")
    print(f"  Multi-county:   {df['multi_county'].sum()}")
    print(f"  Max counties:   {df['num_counties'].max()}")
    
    print(f"\n{'='*80}")
    print(f"✓ Complete! Saved to: {output_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    categorize_all_projects(
        input_csv=input_file,
        output_csv=output_file,
        unique_proposals_file='unique_proposals.txt'
    )
