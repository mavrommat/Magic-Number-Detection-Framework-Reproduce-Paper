import os
import gzip
import shutil
import urllib.request
from vizier_mirrors import get_vizier_candidate_urls, HEADERS


def fetch_and_unzip(url: str, local_path: str, timeout: float = 15.0) -> str | None:
    if not url:
        return local_path if os.path.exists(local_path) else None
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    candidates = get_vizier_candidate_urls(url, include_compression_variants=True)

    for candidate_url in candidates:
        is_gz = candidate_url.endswith(".gz")
        temp_path = local_path + (".gz" if is_gz else ".tmp")

        print(f"Attempting download from {candidate_url}...")
        try:
            req = urllib.request.Request(candidate_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response, open(temp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

            if is_gz:
                print(f"Unzipping {temp_path} to {local_path}...")
                with gzip.open(temp_path, "rb") as f_in, open(local_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(temp_path)
            else:
                os.replace(temp_path, local_path)

            if candidate_url != url:
                print(f"  [Mirror Fallback] Successfully fetched from {candidate_url}")
            return local_path

        except Exception as e:
            print(f"  Download failed for {candidate_url}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    print(f"Error: Exhausted all VizieR mirrors for {url}")
    return None