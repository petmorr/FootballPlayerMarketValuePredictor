"""
Updates player market values by fetching data from an API, merging data, and writing updated files.
"""

import functools
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import ftfy
import pandas as pd
import requests
import unicodedata
from pandas import DataFrame
from rapidfuzz.fuzz import partial_ratio

from logging_config import configure_logger

PREPROC_VARIANT_FOLDERS: Dict[str, Path] = {
    "enhanced_feature_engineering": Path("../data/cleaned/enhanced_feature_engineering"),
    "feature_engineering": Path("../data/cleaned/feature_engineering"),
    "no_feature_engineering": Path("../data/cleaned/no_feature_engineering")
}
UPDATED_VARIANT_FOLDERS: Dict[str, Path] = {
    "enhanced_feature_engineering": Path("../data/updated/enhanced_feature_engineering"),
    "feature_engineering": Path("../data/updated/feature_engineering"),
    "no_feature_engineering": Path("../data/updated/no_feature_engineering")
}
for folder in UPDATED_VARIANT_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

TEMP_UPDATED_FOLDER: Path = Path("../data/updated_temp")
TEMP_UPDATED_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, str] = {"transfermarkt_tld": "co.uk"}
API_BASE_URL: str = "http://localhost:8000"
DATE_FORMAT: str = "%Y-%m-%d"
MAX_API_RETRIES: int = 3

logger = configure_logger("player_value", "player_value.log")


def get_clean_basename(file_path: str) -> str:
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem


def get_updated_filename_from_cleaned(filename: str) -> str:
    base = get_clean_basename(filename)
    suffix = Path(filename).suffix
    if base.startswith("cleaned_"):
        return f"updated_{base[len('cleaned_'):]}{suffix}"
    return f"updated_{base}{suffix}"


def read_input_file(file_path: str) -> Optional[DataFrame]:
    p = Path(file_path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    logger.error(f"Unsupported file type: {file_path}")
    return None


def write_output_file(df: DataFrame, output_folder: Path, original_filename: str) -> None:
    new_filename = get_updated_filename_from_cleaned(original_filename)
    output_path = output_folder / new_filename
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Updated dataset saved to: {output_path}")


def fix_encoding(s: str) -> str:
    try:
        return ftfy.fix_text(s)
    except Exception as e:
        logger.error(f"Encoding fix failed for '{s}': {e}")
        return s


def parse_date(date_str: Any) -> Optional[datetime]:
    if isinstance(date_str, datetime):
        return date_str
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except Exception as e:
        logger.error(f"Date parsing failed for '{date_str}': {e}")
        return None


def remove_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    try:
        name = fix_encoding(name).strip()
        name = remove_diacritics(name)
    except Exception as e:
        logger.error(f"Error normalizing '{name}': {e}")
    name = name.replace("ß", "ss").lower()
    name = re.sub(r"[-']", "", name)
    name = re.sub(r"\s+", "", name)
    return name


def normalize_name_keep_spaces(name: str) -> str:
    try:
        name = fix_encoding(name).strip().lower()
        name = remove_diacritics(name)
    except Exception as e:
        logger.error(f"Error (keep spaces) normalizing '{name}': {e}")
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def get_last_name(full_name: str) -> str:
    parts = full_name.split()
    return normalize_name(parts[-1]) if parts else ""


def generate_candidate_queries(player_name: str) -> set:
    fixed = fix_encoding(player_name).strip()
    variants = set()
    variants.add(fixed)
    variants.add(fixed.lower())
    variants.add(re.sub(r"[^\w\s]", "", fixed))
    variants.add(re.sub(r"[^\w\s]", "", fixed.lower()))
    no_diacritics = remove_diacritics(fixed)
    variants.add(no_diacritics)
    variants.add(no_diacritics.lower())
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics))
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics.lower()))
    hyphen_space = fixed.replace("-", " ")
    variants.add(hyphen_space)
    variants.add(hyphen_space.lower())
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space))
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space.lower()))
    variants.add(normalize_name(player_name))
    variants.add(normalize_name_keep_spaces(player_name))
    parts = fixed.split()
    if len(parts) == 2:
        reversed_order = f"{parts[1]} {parts[0]}"
        variants.add(reversed_order)
        variants.add(reversed_order.lower())
        variants.add(re.sub(r"[^\w\s]", "", reversed_order))
        variants.add(re.sub(r"[^\w\s]", "", reversed_order.lower()))
    if len(parts) >= 3:
        hyphenated = f"{parts[0]}-{parts[1]} " + " ".join(parts[2:])
        variants.add(hyphenated)
        variants.add(hyphenated.lower())
        variants.add(re.sub(r"[^\w\s]", "", hyphenated))
        variants.add(re.sub(r"[^\w\s]", "", hyphenated.lower()))
    if len(parts) >= 2:
        initial_last = f"{parts[0][0]} {parts[-1]}"
        variants.add(initial_last)
        variants.add(initial_last.lower())
        variants.add(re.sub(r"[^\w\s]", "", initial_last))
        variants.add(re.sub(r"[^\w\s]", "", initial_last.lower()))
    if parts:
        last = parts[-1]
        variants.add(last)
        variants.add(last.lower())
        variants.add(re.sub(r"[^\w\s]", "", last))
        variants.add(re.sub(r"[^\w\s]", "", last.lower()))
        variants.add(normalize_name(last))
        variants.add(normalize_name_keep_spaces(last))
    return variants


session = requests.Session()


def make_request_with_retry(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/{endpoint}"
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            logger.debug(f"API call success on attempt {attempt}: {url}")
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"API call failed (attempt {attempt}/{MAX_API_RETRIES}): {e}")
            time.sleep(min(2 ** attempt, 60))
    return {}


def fetch_player_market_value(player_id: str) -> Optional[List[Dict[str, Any]]]:
    endpoint = f"players/{player_id}/market_value"
    logger.info(f"Fetching market value for player ID {player_id}")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])


@functools.lru_cache(maxsize=1000)
def fetch_player_profile(player_id: str) -> Optional[str]:
    endpoint = f"players/{player_id}/profile"
    logger.info(f"Fetching profile URL for player ID {player_id}")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)


def fetch_player_id_by_last_name(player_name: str, team_name: str) -> Optional[str]:
    input_last = get_last_name(player_name)
    if not input_last:
        return None
    endpoint = f"players/search/{input_last}"
    logger.info(f"Fallback search by last name: '{input_last}'")
    search_results = make_request_with_retry(endpoint).get("results", [])
    best_candidate, best_score = None, 0
    for result in search_results:
        candidate_last = get_last_name(result.get("name", ""))
        if candidate_last != input_last:
            continue
        candidate_club = result.get("club", {}).get("name", "")
        score = partial_ratio(team_name.lower(), candidate_club.lower())
        if score > best_score:
            best_score = score
            best_candidate = result
    if best_candidate and best_score >= 80:
        logger.info(f"Fallback found candidate {best_candidate['id']} for '{input_last}', score {best_score}")
        return best_candidate["id"]
    logger.warning(f"No fallback match for '{input_last}' by last name.")
    return None


@functools.lru_cache(maxsize=1000)
def fetch_player_id(player_name: str, team_name: Optional[str] = None) -> Optional[str]:
    fixed_name = fix_encoding(player_name)
    # Direct and punctuation-removed
    for query in [fixed_name, re.sub(r"[^\w\s]", "", fixed_name)]:
        endpoint = f"players/search/{query}"
        logger.info(f"Searching for Player ID: '{query}'")
        results = make_request_with_retry(endpoint).get("results", [])
        if results:
            valid_results = [r for r in results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else results[0]
            logger.info(f"Found Player ID {best_match['id']} for '{query}'")
            return best_match["id"]

    # Alternative hyphen
    alt_name = fixed_name.replace("-", " ")
    if alt_name != fixed_name:
        for query in [alt_name, re.sub(r"[^\w\s]", "", alt_name)]:
            endpoint = f"players/search/{query}"
            logger.info(f"Alt punctuation search: '{query}'")
            results = make_request_with_retry(endpoint).get("results", [])
            if results:
                valid_results = [r for r in results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else results[0]
                logger.info(f"Found Player ID {best_match['id']} for '{query}'")
                return best_match["id"]

    # Hyphenated for multi-part names
    parts = fixed_name.split()
    if len(parts) >= 3:
        hyphenated = f"{parts[0]}-{parts[1]} " + " ".join(parts[2:])
        for query in [hyphenated, re.sub(r"[^\w\s]", "", hyphenated)]:
            endpoint = f"players/search/{query}"
            logger.info(f"Hyphenated search: '{query}'")
            results = make_request_with_retry(endpoint).get("results", [])
            if results:
                valid_results = [r for r in results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else results[0]
                logger.info(f"Found Player ID {best_match['id']} for '{query}'")
                return best_match["id"]

    # Reversed name for two-part
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        for query in [reversed_name, re.sub(r"[^\w\s]", "", reversed_name)]:
            endpoint = f"players/search/{query}"
            logger.info(f"Reversed name search: '{query}'")
            results = make_request_with_retry(endpoint).get("results", [])
            if results:
                valid_results = [r for r in results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else results[0]
                logger.info(f"Found Player ID {best_match['id']} for '{query}'")
                return best_match["id"]
        logger.warning(f"No reversed name results for '{reversed_name}'")

    # Lower-case fallback
    lower_name = fixed_name.lower()
    if lower_name != fixed_name:
        for query in [lower_name, re.sub(r"[^\w\s]", "", lower_name)]:
            endpoint = f"players/search/{query}"
            logger.info(f"Lower-case fallback: '{query}'")
            results = make_request_with_retry(endpoint).get("results", [])
            if results:
                valid_results = [r for r in results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else results[0]
                logger.info(f"Found Player ID {best_match['id']} for '{query}'")
                return best_match["id"]

    # Candidate generator
    candidates = generate_candidate_queries(player_name)
    for query in candidates:
        endpoint = f"players/search/{query}"
        logger.info(f"Candidate search: '{query}'")
        results = make_request_with_retry(endpoint).get("results", [])
        if results:
            valid_results = [r for r in results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else results[0]
            logger.info(f"Found Player ID {best_match['id']} with candidate '{query}'")
            return best_match["id"]

    # Last-name fallback
    if team_name:
        fallback_id = fetch_player_id_by_last_name(fixed_name, team_name)
        if fallback_id:
            return fallback_id

    logger.warning(f"No results for '{fixed_name}' or variants.")
    return None


def slugify(value: str) -> str:
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def construct_transfermarkt_url(player_name: str, player_id: str) -> str:
    tld = CONFIG.get("transfermarkt_tld", "com")
    return f"https://www.transfermarkt.{tld}/{slugify(player_name)}/profil/spieler/{player_id}"


def filter_market_values_by_season(
        market_values: List[Dict[str, Any]],
        season_start: datetime,
        season_end: datetime
) -> List[Dict[str, Any]]:
    filtered = []
    for entry in market_values:
        value_date = parse_date(entry.get("date"))
        if value_date and season_start <= value_date <= season_end:
            filtered.append(entry)
    return filtered


def validate_market_value(
        market_values: List[Dict[str, Any]] | Dict[str, Any],
        team_name: str,
        season_start: datetime,
        season_end: datetime
) -> Optional[Dict[str, Any]]:
    if isinstance(market_values, dict):
        market_values = [market_values]
    season_entries = filter_market_values_by_season(market_values, season_start, season_end)
    closest_entry, closest_diff = None, float("inf")
    for entry in season_entries:
        value_date = parse_date(entry.get("date"))
        if not value_date:
            continue
        club_name = entry.get("clubName", "")
        score = partial_ratio(team_name.lower(), club_name.lower())
        if score >= 80:
            diff = abs((value_date - season_end).days)
            if diff < closest_diff:
                closest_diff = diff
                closest_entry = entry
    if closest_entry:
        return closest_entry
    if market_values:
        try:
            earliest_entry = min(market_values, key=lambda r: parse_date(r.get("date")) or datetime.max)
            logger.info(f"Falling back to earliest market value entry: {earliest_entry}")
            return earliest_entry
        except Exception as e:
            logger.error(f"Error finding earliest entry: {e}")
    logger.warning(f"No valid market value for '{team_name}'.")
    return None


def process_player(player_name: str, team_name: str) -> Optional[List[Dict[str, Any]]]:
    fixed_name = fix_encoding(player_name)
    pid = fetch_player_id(fixed_name, team_name)
    if not pid:
        logger.warning(f"No Player ID for '{fixed_name}'")
        return None
    data = fetch_player_market_value(pid)
    if not data:
        logger.warning(f"No market value data for '{fixed_name}'")
        return None
    return data


def process_player_values(file_path: str, season_end_year: int) -> None:
    df = read_input_file(file_path)
    if df is None:
        return
    df["Market Value"] = None
    season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
    season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)

    def task(idx: int, row: pd.Series) -> Tuple[int, Optional[Any]]:
        player_name = row["player"]
        team_name = row["squad"]
        pid = fetch_player_id(player_name, team_name)
        if not pid:
            return idx, None
        m_values = fetch_player_market_value(pid)
        if not m_values:
            return idx, None
        valid_entry = validate_market_value(m_values, team_name, season_start, season_end)
        if valid_entry and "marketValue" in valid_entry:
            return idx, valid_entry["marketValue"]
        return idx, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(task, i, r) for i, r in df.iterrows()]
        for f in as_completed(futures):
            i, val = f.result()
            if val is not None:
                df.at[i, "Market Value"] = val

    write_output_file(df, TEMP_UPDATED_FOLDER, file_path)


def copy_market_value(source_file: Path, target_file: Path) -> None:
    df_src = pd.read_parquet(source_file)
    df_tgt = pd.read_parquet(target_file)
    if "player" in df_src.columns and "player" in df_tgt.columns:
        df_src["player_norm"] = df_src["player"].apply(normalize_name)
        df_tgt["player_norm"] = df_tgt["player"].apply(normalize_name)
        df_src_unique = df_src.drop_duplicates("player_norm")
        merged = pd.merge(
            df_tgt,
            df_src_unique[["player_norm", "Market Value"]],
            on="player_norm",
            how="left"
        )
        merged.drop(columns=["player_norm"], inplace=True)
    else:
        merged = df_tgt
    merged.to_parquet(target_file, index=False)
    logger.info(f"Merged Market Value into: {target_file}")


def main() -> None:
    enhanced_folder = PREPROC_VARIANT_FOLDERS["enhanced_feature_engineering"]
    enhanced_output = UPDATED_VARIANT_FOLDERS["enhanced_feature_engineering"]
    logger.info(f"Processing enhanced variant in: {enhanced_folder}")
    for file in os.listdir(enhanced_folder):
        if file.endswith(".parquet"):
            file_path = enhanced_folder / file
            match = re.search(r"(\d{4})-(\d{4})", file)
            if not match:
                logger.warning(f"Cannot detect season year in {file}")
                continue
            season_end_year = int(match.group(2))
            try:
                process_player_values(str(file_path), season_end_year)
                updated_file = get_updated_filename_from_cleaned(file)
                temp_file = TEMP_UPDATED_FOLDER / updated_file
                final_file = enhanced_output / updated_file
                if temp_file.is_file():
                    shutil.copy(temp_file, final_file)
                    logger.info(f"Enhanced updated file: {final_file}")
                else:
                    logger.error(f"Enhanced updated file not found: {file}")
            except Exception as e:
                logger.error(f"Error processing enhanced file {file_path}: {e}")

    for variant_key in ["feature_engineering", "no_feature_engineering"]:
        variant_folder = PREPROC_VARIANT_FOLDERS[variant_key]
        output_folder = UPDATED_VARIANT_FOLDERS[variant_key]
        logger.info(f"Processing {variant_key} in: {variant_folder}")
        for file in os.listdir(variant_folder):
            if file.endswith(".parquet"):
                file_path = variant_folder / file
                updated_file = get_updated_filename_from_cleaned(file)
                target_file = output_folder / updated_file
                try:
                    df_variant = pd.read_parquet(file_path)
                    if variant_key == "no_feature_engineering":
                        df_variant = df_variant.drop(columns=["league", "season"], errors="ignore")
                    write_output_file(df_variant, output_folder, file)
                    enhanced_file = UPDATED_VARIANT_FOLDERS["enhanced_feature_engineering"] / updated_file
                    if not enhanced_file.is_file():
                        logger.warning(
                            f"Enhanced file not found for {file}, skipping Market Value merge for {variant_key}."
                        )
                        continue
                    copy_market_value(enhanced_file, target_file)
                    logger.info(f"{variant_key} updated file: {target_file}")
                except Exception as e:
                    logger.error(f"Error in {variant_key} file {file_path}: {e}")

    try:
        for temp_file in TEMP_UPDATED_FOLDER.iterdir():
            temp_file.unlink()
        os.rmdir(TEMP_UPDATED_FOLDER)
        logger.info("Removed temporary folder.")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()