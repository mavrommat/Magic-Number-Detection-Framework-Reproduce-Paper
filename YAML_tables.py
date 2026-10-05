import os
import re
import numpy as np
from astropy.io import ascii
from astropy.io.ascii.core import IntType, FloatType, StrType
from Dowlnload_uzip_catalogs import fetch_and_unzip
from pathlib import Path

catalogs = [
    {
        "name": "J_MNRAS_415_2336_hatlas",
        "readme_url": 'https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/415/2336/ReadMe',
        "data_path": '',
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/415/2336/hatlas.dat.gz"
    },
    {
        "name": "J_ApJ_810_14_table_4",
        "readme_url": 'https://cdsarc.cds.unistra.fr/ftp/J/ApJ/810/14/ReadMe',
        "data_path": '',
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/810/14/table4.dat"
    },
    {
        "name": "J_ApJ_810_14_table_8",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/810/14/ReadMe",
        "data_path": '',
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/810/14/table8.dat"
    },
    {
        "name": "J_A_A_531_A6_tsi_hol",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/531/A6/ReadMe",
        "data_path": '',
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/531/A6/tsi_hol.dat"
    },
    {
        "name": "J_A_A_648_A3_en1_comb",
        "readme_url": "http://cdsarc.u-strasbg.fr/ftp/J/A+A/648/A3/ReadMe",
        "data_path": "",
        "data_url": "http://cdsarc.u-strasbg.fr/ftp/J/A+A/648/A3/en1_comb.dat.gz"
    },
    {
        "name": "J_452_2087_tiling_dat",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/452/2087/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/452/2087/tiling.dat"
    },
    {
        "name": "J_A_A_562_A86_macs1206",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/562/A86/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/562/A86/macs1206.dat.gz"
    },
    {
        "name": "J_A_A_555_A42_main210",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/main210.dat"
    },
    {
        "name": "J_A_A_555_A42_suppl210",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/suppl210.dat"
    },
    {
        "name": "J_A_A_555_A42_main510",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/main510.dat"
    },
    {
        "name": "J_A_A_555_A42_suppl510",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/555/A42/suppl510.dat"
    },
    {
        "name": "ApJ_742_125_xgroups",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/742/125/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/742/125/xgroups.dat"
    },
    {
        "name": "ApJ_742_125_xgal",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/742/125/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/742/125/xgal.dat.gz"
    },
    {
        "name": "ApJ_776_71_table4",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/776/71/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/776/71/table4.dat.gz"
    },
    {
        "name": "VII/269_dr9q",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/VII/269/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/VII/269/dr9q.dat.gz"
    },
    {
        "name": "VII/269_dr9qsup",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/VII/269/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/VII/269/dr9qsup.dat.gz"
    },
    {
        "name": "I_280B_ccp70",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/I/280B/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/I/280B/ccp70.dat.gz"
    },
    {
        "name": "I_280B_ccm00",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/I/280B/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/I/280B/ccm00.dat.gz"
    },
    {
        "name": "J_ApJ_836_99_table5",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/99/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/836/99/table5.dat"
    },
    {
        "name": "J_ApJ_919_51_table2",
        "readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/919/51/ReadMe",
        "data_path": "",
        "data_url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/919/51/table2.dat.gz"
    }
]


#  GENERATOR LOGIC
def safe_string_parser(vals):
    return np.array([str(v).strip() for v in vals], dtype=str)

def generate_yaml(catalog_list):
    output_dir = Path("Real_World_Analysis") / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    local_dir = Path("Real_World_Analysis") / "local_data"
    
    for cat in catalog_list:
        name = cat.get("name", "unnamed_catalog")
        safe_name = name.replace("/", "_").replace("\\", "_")
        
        file_path = output_dir / f"{safe_name}.yaml"
        if file_path.exists():
            print(f" Skipped: {file_path} already exists.")
            continue
            
        print(f" Fetching columns for {name}...")

        expected_local_path = local_dir / f"{safe_name}.dat"
        
        data_url = cat.get('data_url', '')
        readme = cat.get('readme_url', '')

        source = fetch_and_unzip(data_url, str(expected_local_path))

        if not source or not os.path.exists(source):
            print(f" Error: Data missing and no valid URL provided for {name}.")
            continue
        
        converters = {}
        success = False
        column_names = []
        
        while not success:
            try:
                table = ascii.read(source, readme=readme, format="cds", converters=converters)
                column_names = table.colnames
                success = True
                
            except ValueError as e:
                error_str = str(e)
                match = re.search(r"Column\s+(.+?)\s+failed to convert", error_str)
                
                if match:
                    bad_col = match.group(1)
                    if bad_col in converters:
                        print(f" Failed to load {name}. Column '{bad_col}' still failed after applying auto-fix.")
                        break
                        
                    print(f"   [Auto-Fix] Magic number crash detected on column '{bad_col}'. Forcing to string and retrying...")
                    
                    converters[bad_col] = [
                        (safe_string_parser, IntType),
                        (safe_string_parser, FloatType),
                        (safe_string_parser, StrType)
                    ]
                else:
                    print(f" Failed to load {name}. ValueError: {e}")
                    break
                    
            except Exception as e:
                print(f" Failed to load {name}. Error: {e}")
                break

        if not success:
            continue

        # --- YAML CONSTRUCTION ---
        yaml_content = f"""# ==============================================================================
# Magic Number Detection Ground Truth Configuration
# Catalog Name: {name}
# ==============================================================================

catalog_info:
  readme_url: "{readme}"
  data_url: "{data_url}"
  data_path: "{str(expected_local_path)}"
  format: "cds"

# ==============================================================================
# Column Annotations ({len(column_names)} columns detected)
# ==============================================================================
columns:
"""
        for col in column_names:
            yaml_content += f"""  {col}:
    Category: ""
    Ground_Truth_Magic: []
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        print(f" Created: {file_path} (with {len(column_names)} columns auto-filled)")

if __name__ == "__main__":
    print("--- Auto-Extracting Columns and Generating YAMLs ---")
    generate_yaml(catalogs)
    print("--- Done! ---")