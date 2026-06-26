import re
import os

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latex_file = os.path.join(workspace_dir, "latex_plots", "comparison_plots_alie_delta.tex")
    
    print(f"Reading {latex_file}...")
    with open(latex_file, "r") as f:
        content = f.read()
    
    # Find all \includegraphics[...]{filename}
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} image paths. Checking exist status...")
    
    missing_files = []
    for path in matches:
        abs_path = os.path.join(workspace_dir, path)
        exists = os.path.exists(abs_path)
        print(f"[{'OK' if exists else 'MISSING'}] {path}")
        if not exists:
            missing_files.append(path)
            
    print("\nSummary:")
    if missing_files:
        print(f"ERROR: {len(missing_files)} file(s) are missing!")
        for f in missing_files:
            print(f" - {f}")
        exit(1)
    else:
        print("All image files exist successfully!")
        exit(0)

if __name__ == "__main__":
    main()
