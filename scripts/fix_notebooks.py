"""
Script to fix Colab dependencies and hardcoded paths in Jupyter notebooks.

This script:
1. Removes Google Colab imports and drive.mount() calls
2. Replaces hardcoded paths with relative paths
3. Adds proper path setup for local execution

Author: Mritunjay Kumar
"""

import json
import os
import re
from pathlib import Path

# Path patterns to replace
PATH_REPLACEMENTS = [
    (r"//content/drive/My Drive/jubilant/jubilant/", "../../data/"),
    (r"//content/drive/My Drive/jubilant/", "../../outputs/"),
    (r'/content/drive/My Drive/jubilant/jubilant/', '../../data/'),
    (r'/content/drive/My Drive/jubilant/', '../../outputs/'),
    (r'//content/drive/My Drive/ML_AI/DeepFuture/', '../../src/'),
]

# Colab-specific imports to remove
COLAB_PATTERNS = [
    r'from google\.colab import drive',
    r'import.*google\.colab',
    r"drive\.mount\(['\"].*['\"].*\)",
]


def fix_notebook(notebook_path: str, dry_run: bool = False):
    """
    Fix a single notebook file.
    
    Args:
        notebook_path: Path to notebook file
        dry_run: If True, only print changes without modifying
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(notebook_path)}")
    print(f"{'='*60}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes = 0
    cells_modified = 0
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            original_source = cell['source']
            modified_source = []
            cell_changed = False
            
            for line in original_source:
                modified_line = line
                
                # Remove Colab imports
                for pattern in COLAB_PATTERNS:
                    if re.search(pattern, line):
                        print(f"  ❌ Removing: {line.strip()}")
                        modified_line = ''
                        cell_changed = True
                        changes += 1
                        break
                
                # Replace hardcoded paths
                if modified_line:
                    for old_path, new_path in PATH_REPLACEMENTS:
                        if old_path in modified_line:
                            new_line = modified_line.replace(old_path, new_path)
                            print(f"  🔄 Path fix: {old_path} → {new_path}")
                            modified_line = new_line
                            cell_changed = True
                            changes += 1
                
                if modified_line:  # Only add non-empty lines
                    modified_source.append(modified_line)
            
            if cell_changed:
                cells_modified += 1
                cell['source'] = modified_source
    
    # Add setup cell at the beginning if needed
    if changes > 0:
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup paths for local execution\n",
                "import os\n",
                "import sys\n",
                "\n",
                "# Project directories\n",
                "BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "DATA_DIR = os.path.join(BASE_DIR, 'data')\n",
                "OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')\n",
                "SRC_DIR = os.path.join(BASE_DIR, 'src')\n",
                "\n",
                "# Add src to path\n",
                "sys.path.insert(0, SRC_DIR)\n"
            ]
        }
        
        # Insert setup cell after the first cell (usually imports)
        if len(notebook['cells']) > 0:
            notebook['cells'].insert(1, setup_cell)
            print(f"  ✅ Added path setup cell")
    
    print(f"\n  Summary: {changes} changes in {cells_modified} cells")
    
    if not dry_run and changes > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  💾 Saved changes to file")
    elif dry_run:
        print(f"  🔍 DRY RUN - No changes written")
    
    return changes


def fix_all_notebooks(directory: str, dry_run: bool = False):
    """
    Fix all notebooks in a directory.
    
    Args:
        directory: Directory containing notebooks
        dry_run: If True, only print changes without modifying
    """
    notebook_dir = Path(directory)
    notebooks = list(notebook_dir.glob('*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks in {directory}")
    
    total_changes = 0
    for nb_path in notebooks:
        changes = fix_notebook(str(nb_path), dry_run=dry_run)
        total_changes += changes
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_changes} changes across {len(notebooks)} notebooks")
    print(f"{'='*60}")
    
    return total_changes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Colab dependencies in notebooks')
    parser.add_argument('directory', help='Directory containing notebooks')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show changes without modifying files')
    
    args = parser.parse_args()
    
    fix_all_notebooks(args.directory, dry_run=args.dry_run)
