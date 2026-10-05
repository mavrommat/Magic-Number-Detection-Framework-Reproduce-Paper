import os
import re
import gzip
import shutil
import urllib.request
import numpy as np
from astropy.io import ascii
from astropy.io.ascii.core import IntType, FloatType, StrType

def fetch_and_unzip(url, local_path):
    if not url: 
        return local_path
    if os.path.exists(local_path): 
        return local_path

    # Ensure the target local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    is_gz = url.endswith('.gz')
    temp_path = local_path + (".gz" if is_gz else "")

    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, temp_path)
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return None

    if is_gz:
        print(f"Unzipping to {local_path}...")
        with gzip.open(temp_path, 'rb') as f_in:
            with open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(temp_path)

    return local_path