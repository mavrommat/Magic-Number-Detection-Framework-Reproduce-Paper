from __future__ import annotations
import re
import urllib.request
import urllib.error
import time
from pathlib import Path
import shutil


# Ordered with non-.fr international mirrors prioritized first while .fr is down
VIZIER_MIRROR_BASES = [
    # Harvard CfA (USA)
    "https://vizier.cfa.harvard.edu/vizier/ftp/cats",
    "https://vizier.cfa.harvard.edu/ftp/cats",
    # NAOJ / ADAC (Japan)
    "https://vizier.nao.ac.jp/vizier/ftp/cats",
    "https://vizier.nao.ac.jp/ftp/cats",
    # IUCAA (India)
    "https://vizier.iucaa.in/vizier/ftp/cats",
    "http://vizier.iucaa.in/ftp/cats",
    # BAO / NAOC (China)
    "http://vizier.nao.cas.cn/vizier/ftp/cats",
    "http://vizier.nao.cas.cn/ftp/cats",
    # SAAO (South Africa)
    "https://vizier.saao.ac.za/vizier/ftp/cats",
    # Primary Strasbourg (HTTPS & legacy HTTP) - kept at bottom during .fr outage
    "https://cdsarc.cds.unistra.fr/ftp",
    "http://cdsarc.u-strasbg.fr/ftp",
]

_CDS_PATH_RE = re.compile(
    r"^https?://[^/]+/(?:vizier/)?ftp/(?:cats/)?(?P<rel_path>.+)$",
    re.IGNORECASE,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Astropy/VizieR-Mirror-Fallback)"}


def get_vizier_candidate_urls(url: str, include_compression_variants: bool = False) -> list[str]:

    if not url or not str(url).startswith(("http://", "https://", "ftp://")):
        return [url]

    match = _CDS_PATH_RE.match(str(url).strip())
    if not match:
        return [url]

    rel_path = match.group("rel_path").lstrip("/")
    rel_paths = [rel_path]

    # .dat uncompressed and .dat.gz
    if include_compression_variants and not rel_path.endswith("ReadMe"):
        if rel_path.endswith(".gz"):
            rel_paths.append(rel_path[:-3])
        else:
            rel_paths.append(rel_path + ".gz")

    candidates: list[str] = []
    for rp in rel_paths:
        for base in VIZIER_MIRROR_BASES:
            candidate = f"{base.rstrip('/')}/{rp}"
            if candidate not in candidates:
                candidates.append(candidate)

    if url not in candidates:
        candidates.append(url)

    # try all  mirros and last .fr
    candidates.sort(key=lambda u: (".unistra.fr" in u or ".u-strasbg.fr" in u))
    return candidates


def resolve_working_vizier_url(
    url: str,
    per_mirror_timeout: float = 2.5,
    max_total_time: float = 8.0,
    include_compression_variants: bool = False,
) -> str:
    
    if not url or not str(url).startswith(("http://", "https://")):
        return url

    candidates = get_vizier_candidate_urls(
        url, include_compression_variants=include_compression_variants
    )
    errors: list[str] = []
    start_time = time.monotonic()

    for candidate in candidates:
        elapsed = time.monotonic() - start_time
        remaining = max_total_time - elapsed
        if remaining <= 0:
            errors.append(f"Total search timeout ({max_total_time:.1f}s) exceeded")
            break

        current_timeout = min(per_mirror_timeout, remaining)

        req = urllib.request.Request(candidate, method="HEAD", headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=current_timeout) as resp:
                if 200 <= resp.status < 400:
                    if candidate != url:
                        print(f"  [Mirror Fallback] Routed {url} -> {candidate}")
                    return candidate
        except urllib.error.HTTPError as e:
            if e.code in (403, 405):
                # Recompute remaining time before the fallback 1-byte GET
                remaining = max_total_time - (time.monotonic() - start_time)
                if remaining <= 0:
                    errors.append(f"Total search timeout ({max_total_time:.1f}s) exceeded")
                    break
                try:
                    get_headers = {**HEADERS, "Range": "bytes=0-0"}
                    get_req = urllib.request.Request(candidate, headers=get_headers)
                    with urllib.request.urlopen(
                        get_req, timeout=min(per_mirror_timeout, remaining)
                    ) as resp:
                        if 200 <= resp.status < 400:
                            if candidate != url:
                                print(f"  [Mirror Fallback] Routed {url} -> {candidate}")
                            return candidate
                except Exception as inner_e:
                    errors.append(f"{candidate} ({type(inner_e).__name__}: {inner_e})")
            else:
                errors.append(f"{candidate} (HTTP {e.code})")
        except Exception as e:
            errors.append(f"{candidate} ({type(e).__name__}: {e})")

    raise RuntimeError(
        f"Could not resolve '{url}' within {max_total_time}s. Tried:\n  - "
        + "\n  - ".join(errors)
    )


def fetch_readme_with_timeout(
    readme_url: str,
    local_readme_path: str | Path | None = None,
    per_mirror_timeout: float = 2.5,
    max_total_time: float = 8.0,
) -> str:
    
    if local_readme_path:
        local_path = Path(local_readme_path)
        if local_path.is_file() and local_path.stat().st_size > 0:
            return str(local_path)

    working_url = resolve_working_vizier_url(
        readme_url,
        per_mirror_timeout=per_mirror_timeout,
        max_total_time=max_total_time,
    )

    if local_readme_path:
        local_path = Path(local_readme_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(working_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=per_mirror_timeout) as resp, open(
            local_path, "wb"
        ) as out:
            shutil.copyfileobj(resp, out)
        return str(local_path)

    return working_url