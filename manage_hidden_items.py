#!/usr/bin/env python3
"""
Utility to manage hidden items configuration for model training
Allows adding/removing items from the filtering configuration
"""

import json
import argparse
import re

def load_config(config_file='hidden_items_config.json'):
    """Load the current hidden items configuration"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found!")
        return None

def save_config(config, config_file='hidden_items_config.json'):
    """Save the updated configuration"""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_file}")

def add_pattern_exclusion(config, category, pattern, reason, description=""):
    """Add a new pattern exclusion rule"""
    if 'filtering_rules' not in config:
        config['filtering_rules'] = {}
    if category not in config['filtering_rules']:
        config['filtering_rules'][category] = {'exclude_patterns': [], 'exclude_specific_items': []}
    
    new_rule = {
        "pattern": pattern,
        "reason": reason,
        "description": description
    }
    
    config['filtering_rules'][category].setdefault('exclude_patterns', []).append(new_rule)
    print(f"Added pattern exclusion: {pattern} for category {category}")

def add_item_exclusion(config, category, items, reason, description=""):
    """Add specific items to exclusion list"""
    if 'filtering_rules' not in config:
        config['filtering_rules'] = {}
    if category not in config['filtering_rules']:
        config['filtering_rules'][category] = {'exclude_patterns': [], 'exclude_specific_items': []}
    
    new_rule = {
        "items": items if isinstance(items, list) else [items],
        "reason": reason,
        "description": description
    }
    
    config['filtering_rules'][category].setdefault('exclude_specific_items', []).append(new_rule)
    print(f"Added item exclusion: {items} for category {category}")

def remove_pattern_exclusion(config, category, pattern):
    """Remove a pattern exclusion rule"""
    try:
        rules = config['filtering_rules'][category]['exclude_patterns']
        config['filtering_rules'][category]['exclude_patterns'] = [
            rule for rule in rules if rule['pattern'] != pattern
        ]
        print(f"Removed pattern exclusion: {pattern} from category {category}")
    except KeyError:
        print(f"Pattern {pattern} not found in category {category}")

def list_exclusions(config, category=None):
    """List all current exclusion rules"""
    if category:
        categories = [category]
    else:
        categories = config.get('filtering_rules', {}).keys()
    
    for cat in categories:
        if cat in config.get('filtering_rules', {}):
            rules = config['filtering_rules'][cat]
            print(f"\n=== {cat.upper()} ===")
            
            print("Pattern Exclusions:")
            for rule in rules.get('exclude_patterns', []):
                print(f"  Pattern: {rule['pattern']}")
                print(f"  Reason: {rule['reason']}")
                if rule.get('description'):
                    print(f"  Description: {rule['description']}")
                print()
            
            print("Specific Item Exclusions:")
            for rule in rules.get('exclude_specific_items', []):
                print(f"  Items: {rule['items']}")
                print(f"  Reason: {rule['reason']}")
                if rule.get('description'):
                    print(f"  Description: {rule['description']}")
                print()

def main():
    parser = argparse.ArgumentParser(description='Manage hidden items configuration')
    parser.add_argument('--config', default='hidden_items_config.json', help='Configuration file path')
    parser.add_argument('--list', action='store_true', help='List current exclusions')
    parser.add_argument('--category', help='Category to operate on (e.g., 716_pavement_markings)')
    parser.add_argument('--add-pattern', help='Add pattern exclusion')
    parser.add_argument('--add-items', nargs='+', help='Add specific item exclusions')
    parser.add_argument('--reason', help='Reason for exclusion')
    parser.add_argument('--description', help='Detailed description')
    parser.add_argument('--remove-pattern', help='Remove pattern exclusion')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return
    
    # List exclusions
    if args.list:
        list_exclusions(config, args.category)
        return
    
    # Add pattern exclusion
    if args.add_pattern:
        if not args.category or not args.reason:
            print("Error: --category and --reason required for adding patterns")
            return
        add_pattern_exclusion(config, args.category, args.add_pattern, args.reason, args.description or "")
        save_config(config, args.config)
        return
    
    # Add item exclusions
    if args.add_items:
        if not args.category or not args.reason:
            print("Error: --category and --reason required for adding items")
            return
        add_item_exclusion(config, args.category, args.add_items, args.reason, args.description or "")
        save_config(config, args.config)
        return
    
    # Remove pattern exclusion
    if args.remove_pattern:
        if not args.category:
            print("Error: --category required for removing patterns")
            return
        remove_pattern_exclusion(config, args.category, args.remove_pattern)
        save_config(config, args.config)
        return
    
    # Default: list current config
    list_exclusions(config)

if __name__ == "__main__":
    main()