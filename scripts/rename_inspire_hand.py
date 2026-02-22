#!/usr/bin/env python3
"""Script to rename inspire_hand rigid object directories and update USD references.

For each rigid object:
- <name>   -> visuals
- <name>_1 -> collisions (with reference path updated to ./collisions.usda and prim_path without _1)
"""

import os
import re
from pathlib import Path

# Try to import USD - this script should be run with Isaac Sim Python
try:
    from pxr import Usd, Sdf
except ImportError:
    print("ERROR: This script must be run with Isaac Sim Python:")
    print("  ./isaaclab.sh -p scripts/rename_inspire_hand.py")
    exit(1)


def find_and_update_references_in_usd(usd_file: Path, old_path_pattern: str, new_path: str, 
                                       old_prim_pattern: str, new_prim_suffix: str):
    """Find and update references in a USD file."""
    if not usd_file.exists():
        return False
    
    try:
        stage = Usd.Stage.Open(str(usd_file))
        if stage is None:
            print(f"  WARNING: Could not open {usd_file}")
            return False
        
        modified = False
        
        # Iterate through all prims
        for prim in stage.Traverse():
            # Check for references
            refs = prim.GetReferences()
            refs_list = refs.GetAddedOrExplicitItems()
            
            for ref in refs_list:
                asset_path = ref.assetPath
                prim_path = ref.primPath.pathString if ref.primPath else ""
                
                # Check if this reference matches our pattern
                if old_path_pattern in asset_path:
                    # Update the reference
                    new_asset_path = asset_path.replace(old_path_pattern, new_path)
                    new_prim_path = prim_path.replace("_1", "") if prim_path else ""
                    
                    print(f"  Updating reference in {prim.GetPath()}:")
                    print(f"    Asset: {asset_path} -> {new_asset_path}")
                    if prim_path:
                        print(f"    Prim:  {prim_path} -> {new_prim_path}")
                    
                    # Remove old and add new reference
                    refs.RemoveReference(ref)
                    if new_prim_path:
                        refs.AddReference(Sdf.Reference(new_asset_path, Sdf.Path(new_prim_path)))
                    else:
                        refs.AddReference(Sdf.Reference(new_asset_path))
                    modified = True
        
        if modified:
            stage.Save()
            print(f"  Saved {usd_file}")
        
        return modified
    
    except Exception as e:
        print(f"  ERROR processing {usd_file}: {e}")
        return False


def process_rigid_object_dir(rigid_obj_dir: Path):
    """Process a single rigid object directory."""
    print(f"\nProcessing: {rigid_obj_dir}")
    
    # Find all subdirectories
    subdirs = [d for d in rigid_obj_dir.iterdir() if d.is_dir()]
    
    # Separate into _1 suffix and non-_1 suffix
    dirs_with_1 = [d for d in subdirs if d.name.endswith("_1")]
    dirs_without_1 = [d for d in subdirs if not d.name.endswith("_1") and d.name not in ["visuals", "collisions"]]
    
    # Match pairs
    for dir_with_1 in dirs_with_1:
        base_name = dir_with_1.name[:-2]  # Remove "_1"
        matching_dir = rigid_obj_dir / base_name
        
        if matching_dir.exists() and matching_dir in dirs_without_1:
            print(f"  Found pair: {base_name} and {dir_with_1.name}")
            
            visuals_dir = rigid_obj_dir / "visuals"
            collisions_dir = rigid_obj_dir / "collisions"
            
            # Rename directories
            if not visuals_dir.exists():
                print(f"  Renaming {base_name} -> visuals")
                matching_dir.rename(visuals_dir)
            else:
                print(f"  WARNING: visuals already exists, skipping {base_name}")
            
            if not collisions_dir.exists():
                print(f"  Renaming {dir_with_1.name} -> collisions")
                dir_with_1.rename(collisions_dir)
            else:
                print(f"  WARNING: collisions already exists, skipping {dir_with_1.name}")
    
    # Also handle unpaired directories (just _1 without matching base)
    for dir_with_1 in dirs_with_1:
        base_name = dir_with_1.name[:-2]
        matching_dir = rigid_obj_dir / base_name
        
        if not matching_dir.exists() and dir_with_1.exists():
            collisions_dir = rigid_obj_dir / "collisions"
            if not collisions_dir.exists():
                print(f"  Renaming unpaired {dir_with_1.name} -> collisions")
                dir_with_1.rename(collisions_dir)


def update_references_in_parent_usd(rigid_obj_dir: Path):
    """Update references in the parent USD file."""
    # Look for USD files in the parent directory that might reference this
    parent_dir = rigid_obj_dir.parent
    
    for usd_file in rigid_obj_dir.glob("*.usd*"):
        print(f"\nChecking {usd_file} for references to update...")
        
        try:
            stage = Usd.Stage.Open(str(usd_file))
            if stage is None:
                continue
            
            modified = False
            root_layer = stage.GetRootLayer()
            
            # Check sublayers
            sublayers = list(root_layer.subLayerPaths)
            new_sublayers = []
            for sublayer in sublayers:
                new_path = sublayer
                # Update paths ending with _1 to collisions
                if "_1/" in sublayer or "_1." in sublayer:
                    # Extract the base name and replace
                    new_path = re.sub(r'/([^/]+)_1/', r'/collisions/', sublayer)
                    new_path = re.sub(r'/([^/]+)_1\.', r'/collisions.', new_path)
                    if new_path != sublayer:
                        print(f"  Sublayer: {sublayer} -> {new_path}")
                        modified = True
                new_sublayers.append(new_path)
            
            if modified:
                root_layer.subLayerPaths = new_sublayers
            
            # Check references in prims
            for prim in stage.Traverse():
                refs = prim.GetReferences()
                refs_list = list(refs.GetAddedOrExplicitItems())
                
                for ref in refs_list:
                    asset_path = ref.assetPath
                    prim_path = ref.primPath.pathString if ref.primPath else ""
                    
                    new_asset_path = asset_path
                    new_prim_path = prim_path
                    
                    # Update asset path if it contains _1
                    if "_1/" in asset_path or "_1." in asset_path:
                        new_asset_path = re.sub(r'/([^/]+)_1/', r'/collisions/', asset_path)
                        new_asset_path = re.sub(r'/([^/]+)_1\.', r'/collisions.', new_asset_path)
                    
                    # Update prim path if it contains _1
                    if "_1" in prim_path:
                        new_prim_path = prim_path.replace("_1", "")
                    
                    if new_asset_path != asset_path or new_prim_path != prim_path:
                        print(f"  Reference in {prim.GetPath()}:")
                        print(f"    Asset: {asset_path} -> {new_asset_path}")
                        print(f"    Prim:  {prim_path} -> {new_prim_path}")
                        
                        refs.RemoveReference(ref)
                        if new_prim_path:
                            refs.AddReference(Sdf.Reference(new_asset_path, Sdf.Path(new_prim_path)))
                        else:
                            refs.AddReference(Sdf.Reference(new_asset_path))
                        modified = True
            
            if modified:
                stage.Save()
                print(f"  Saved {usd_file}")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    # Path to inspire_hand directory
    inspire_hand_dir = Path("/home/zhengyuz/Projects/isaaclab/source/isaaclab_assets/data/Robots/inspire")
    
    if not inspire_hand_dir.exists():
        print(f"ERROR: Directory not found: {inspire_hand_dir}")
        return
    
    print(f"Processing inspire hand directory: {inspire_hand_dir}")
    
    # Find all rigid object directories (directories that contain _1 subdirs)
    for subdir in inspire_hand_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Check if this directory has _1 pattern subdirs
        has_1_pattern = any(d.name.endswith("_1") for d in subdir.iterdir() if d.is_dir())
        
        if has_1_pattern:
            process_rigid_object_dir(subdir)
            update_references_in_parent_usd(subdir)
    
    # Also check the main inspire_hand directory itself
    has_1_pattern = any(d.name.endswith("_1") for d in inspire_hand_dir.iterdir() if d.is_dir())
    if has_1_pattern:
        process_rigid_object_dir(inspire_hand_dir)
        update_references_in_parent_usd(inspire_hand_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
