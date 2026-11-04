"""
Feature pipeline for the 2014-15 subscription-renewal Kaggle task.

Usage:
    from src.features import load_tables, build_features
    T = load_tables()                       # reads data/raw/*.csv
    X_train = build_features(train["account.id"], T)
    X_test  = build_features(test["ID"], T)

Everything is computed per account from history only (seasons <= 2013-14),
so train and test go through identical code. No label information is used here.

Cleaning decisions (approved):
  * first.donated kept as-is (incl. 2014 dates)
  * donation amounts: log1p, no capping
  * billing zip: first 5 digits, zero-padded, unmatched/foreign -> NaN, plus zip_missing flag.
    Geo columns (GEO_COLS) are left NaN -> impute inside the CV pipeline for linear models.
  * subscription `section` dropped as a "latest value" feature (100% null in 2013-14);
    instead the last non-null section is used ("last known section").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LAST_SEASON = 2013          # start year of last observed season (2013-14)
LABEL_SEASON = 2014         # season we are predicting (2014-15)
REF_DATE = pd.Timestamp("2014-07-01")
NO_HISTORY = 25             # sentinel for "seasons since last X" when there is no X

# --------------------------------------------------------------------------- #
# Composer -> era dictionary (surnames as they appear in ALL CAPS in concerts.what)
# Review/edit freely: era assignment is approximate.
# --------------------------------------------------------------------------- #
COMPOSER_ERA = {
    # Baroque
    "HANDEL": "baroque", "VIVALDI": "baroque", "BACH": "baroque", "CORELLI": "baroque",
    "TELEMANN": "baroque", "PERGOLESI": "baroque", "DURANTE": "baroque", "RAMEAU": "baroque",
    "GEMINIANI": "baroque", "SCARLATTI": "baroque", "SCHMELZER": "baroque", "LOCKE": "baroque",
    "PURCELL": "baroque", "ARNE": "baroque", "LAWES": "baroque", "ZAVATERI": "baroque",
    "REBEL": "baroque", "FASCH": "baroque", "PORPORA": "baroque", "BROSCHI": "baroque",
    "MOSSI": "baroque", "LOCATELLI": "baroque", "MUFFAT": "baroque", "SCHEIN": "baroque",
    "BIBER": "baroque", "UMSTATT": "baroque", "PISENDEL": "baroque", "ZELENKA": "baroque",
    "VEJVANOVSKY": "baroque", "STANLEY": "baroque", "CROFT": "baroque", "BOYCE": "baroque",
    "SAMMARTINI": "baroque", "BOCCHERINI": "classical",
    # Classical
    "MOZART": "classical", "HAYDN": "classical", "BEETHOVEN": "classical", "GLUCK": "classical",
    "HUMMEL": "classical", "BORTNIANSKY": "classical", "BEREZOVSKY": "classical",
    "FOMIN": "classical", "FACIUS": "classical", "ESTERH": "classical",
    # Romantic
    "MENDELSSOHN": "romantic", "SCHUMANN": "romantic", "BRAHMS": "romantic",
    "SPOHR": "romantic", "GLINKA": "romantic", "ROSSINI": "romantic", "CHOPIN": "romantic",
    # Modern / contemporary
    "LEVIN": "modern", "STOOKEY": "modern",
}
COMPOSERS = sorted(COMPOSER_ERA)
ERAS = ["baroque", "classical", "romantic", "modern"]
KEY_COMPOSERS = ["BACH", "HANDEL", "VIVALDI", "MOZART", "HAYDN", "BEETHOVEN", "TELEMANN", "CORELLI"]

VOCAL_RE = re.compile(
    r"soprano|mezzo|tenor|baritone|countertenor|\bbass\b|\balto\b|chorale|chorus|choir|"
    r"cantata|\bmass\b|missa|messiah|oratorio|requiem|\barias?\b|dixit|vocal|singer|"
    r"voices?|opera", re.I)
HOLIDAY_RE = re.compile(r"messiah|christmas|holiday|nativity|noel|joyous", re.I)
TENTPOLE_RE = re.compile(r"messiah|four seasons|brandenburg|creation|mass in b|"
                         r"water music|magnificat|st\.? matthew|st\.? john passion", re.I)
NON_SOLOIST_RE = re.compile(r"conductor|director|chorale|orchestra|ensemble", re.I)

# Approximate venue coordinates (lat, lon) for distance features
VENUES = {
    "sf": (37.7778, -122.4202),          # SF (Davies Hall area)
    "berkeley": (37.8680, -122.2600),    # Berkeley (campus / church venues)
    "peninsula": (37.4320, -122.1700),   # Peninsula (Palo Alto / Stanford area)
    "walnutcreek": (37.9020, -122.0650), # Contra Costa (Walnut Creek)
}

SUB_PACKAGES = ["Full", "Quartet", "Trio", "Other"]
SUB_LOCATIONS = ["San Francisco", "Berkeley Sunday", "Berkeley Saturday", "Peninsula",
                 "Contra Costa", "Santa Rosa", "Orange County"]
SUB_SECTIONS = ["Orchestra", "Balcony Front", "Premium Orchestra", "Balcony", "Dress Circle",
                "Balcony Rear", "Orchestra Front", "Box"]
TK_LOCATIONS = ["San Francisco", "Berkeley Saturday", "Berkeley Sunday", "Peninsula"]

GEO_COLS = ["geo_lat", "geo_lon", "geo_log_pop", "geo_log_wages", "geo_wage_per_return",
            "dist_sf", "dist_berkeley", "dist_peninsula", "dist_walnutcreek", "dist_min"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    d = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / "raw"
    zc = pd.read_csv(d / "zipcodes.csv", dtype={"Zipcode": str})
    zc["Zipcode"] = zc["Zipcode"].str.zfill(5)          # table drops leading zeros
    return {
        "account": pd.read_csv(d / "account.csv", encoding="latin-1",
                               dtype={"shipping.zip.code": str, "billing.zip.code": str},
                               parse_dates=["first.donated"]),
        "subscriptions": pd.read_csv(d / "subscriptions.csv"),
        "tickets": pd.read_csv(d / "tickets_all.csv"),
        "concerts": pd.read_csv(d / "concerts.csv", encoding="utf-8-sig"),
        "concerts_1415": pd.read_csv(d / "concerts_2014-15.csv", encoding="utf-8-sig"),
        "zipcodes": zc,
    }


def season_start(s: pd.Series) -> pd.Series:
    return s.str[:4].astype(int)


def _onehot(values: pd.Series, categories: list[str], prefix: str, other: bool = True) -> pd.DataFrame:
    v = values.where(values.isin(categories), "Other" if other else np.nan)
    out = pd.DataFrame({f"{prefix}_{c}": (v == c).astype(int) for c in categories}, index=values.index)
    if other:
        out[f"{prefix}_Other"] = (v == "Other").astype(int)
    return out


# --------------------------------------------------------------------------- #
# Subscription features
# --------------------------------------------------------------------------- #
def subscription_features(ids: pd.Index, sub: pd.DataFrame) -> pd.DataFrame:
    s = sub.copy()
    s["start"] = season_start(s["season"])
    s["pkg_family"] = np.select(
        [s["package"].fillna("").str.startswith("Full"), s["package"].fillna("").str.startswith("Quartet"),
         s["package"].fillna("").str.startswith("Trio")], ["Full", "Quartet", "Trio"], "Other")
    s = s.sort_values(["account.id", "start"])

    rows = {}
    for acct, g in s.groupby("account.id", sort=False):
        yrs = g["start"].to_numpy()
        yset = set(yrs)
        streak = 0
        y = LAST_SEASON
        while y in yset:
            streak += 1
            y -= 1
        # longest consecutive run + largest gap
        longest, run = 1, 1
        for a, b in zip(yrs[:-1], yrs[1:]):
            run = run + 1 if b == a + 1 else 1
            longest = max(longest, run)
        gaps = np.diff(yrs) - 1 if len(yrs) > 1 else np.array([0])
        n_last3 = len(yset & {LAST_SEASON - 2, LAST_SEASON - 1, LAST_SEASON})
        pl = g["price.level"].dropna().to_numpy()
        seats = g["no.seats"].to_numpy()
        sec = g["section"].dropna()
        rows[acct] = {
            "sub_n_seasons": len(yrs),
            "sub_seasons_since_last": LAST_SEASON - yrs[-1],
            "sub_years_since_first": LABEL_SEASON - yrs[0],
            "sub_current_streak": streak,
            "sub_longest_streak": longest,
            "sub_longest_gap": int(gaps.max()),
            "sub_lapsed_returned": int((gaps > 0).any()),
            "sub_n_last3": n_last3,
            "sub_frac_last3": n_last3 / len(yrs),
            "sub_price_level_last": pl[-1] if len(pl) else -1,
            "sub_price_level_change": (pl[-1] - pl[-2]) if len(pl) > 1 else 0,
            "sub_price_level_mean": pl.mean() if len(pl) else -1,
            "sub_tier_last": g["subscription_tier"].iloc[-1],
            "sub_seats_last": seats[-1],
            "sub_seats_change": (seats[-1] - seats[-2]) if len(seats) > 1 else 0,
            "sub_seats_mean": seats.mean(),
            "sub_multiple_last": int(g["multiple.subs"].iloc[-1] == "yes"),
            "_pkg": g["pkg_family"].iloc[-1],
            "_loc": g["location"].iloc[-1] if pd.notna(g["location"].iloc[-1]) else "Other",
            "_sec": sec.iloc[-1] if len(sec) else "None",
        }
    f = pd.DataFrame.from_dict(rows, orient="index").reindex(ids)
    ever = f["sub_n_seasons"].notna()
    f.insert(0, "sub_ever", ever.astype(int))
    f.insert(1, "sub_subscribed_2013", (f["sub_seasons_since_last"] == 0).astype(int))
    fills = {"sub_n_seasons": 0, "sub_seasons_since_last": NO_HISTORY, "sub_years_since_first": 0,
             "sub_current_streak": 0, "sub_longest_streak": 0, "sub_longest_gap": 0,
             "sub_lapsed_returned": 0, "sub_n_last3": 0, "sub_frac_last3": 0.0,
             "sub_price_level_last": -1, "sub_price_level_change": 0, "sub_price_level_mean": -1,
             "sub_tier_last": 0, "sub_seats_last": 0, "sub_seats_change": 0, "sub_seats_mean": 0,
             "sub_multiple_last": 0}
    f = f.fillna(fills)
    pkg = f.pop("_pkg").where(ever, "None")
    loc = f.pop("_loc").where(ever, "None")
    sec = f.pop("_sec").where(ever, "None")
    f = pd.concat([f,
                   _onehot(pkg, SUB_PACKAGES, "sub_pkg", other=False),
                   _onehot(loc, SUB_LOCATIONS, "sub_loc"),
                   _onehot(sec, SUB_SECTIONS, "sub_lastsec")], axis=1)
    return f


# --------------------------------------------------------------------------- #
# Donation / account features
# --------------------------------------------------------------------------- #
def donation_features(ids: pd.Index, acc: pd.DataFrame) -> pd.DataFrame:
    a = acc.set_index("account.id").reindex(ids)
    life, y13, n = a["amount.donated.lifetime"], a["amount.donated.2013"], a["no.donations.lifetime"]
    f = pd.DataFrame(index=ids)
    f["don_donor"] = (n > 0).astype(int)
    f["don_gave_2013"] = (y13 > 0).astype(int)
    f["don_2013_log"] = np.log1p(y13)
    f["don_lifetime_log"] = np.log1p(life)
    f["don_n"] = n
    f["don_avg_gift_log"] = np.log1p(life / n.replace(0, np.nan)).fillna(0)
    f["don_share_2013"] = (y13 / life.replace(0, np.nan)).fillna(0)
    f["don_years_since_first"] = ((REF_DATE - a["first.donated"]).dt.days / 365.25).fillna(0)
    f["rel_present"] = a["relationship"].notna().astype(int)
    return f


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def clean_zip(z: pd.Series) -> pd.Series:
    """First run of digits -> first 5 chars, zero-padded. Foreign/non-numeric -> NaN."""
    z = z.astype("string").str.strip()
    ok = z.str.match(r"^\d{3,5}(-?\d{4})?$").fillna(False)     # 3-5 digits, optional +4
    d = z.where(ok).str.extract(r"^(\d+)")[0]
    d = d.where(d.str.len() <= 5, d.str[:5])                    # 9-digit zip without dash
    return d.str.zfill(5)


def _haversine(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def geo_features(ids: pd.Index, acc: pd.DataFrame, zc: pd.DataFrame) -> pd.DataFrame:
    a = acc.set_index("account.id").reindex(ids)
    zip5 = clean_zip(a["billing.zip.code"].fillna(a["shipping.zip.code"]))
    z = zc.drop_duplicates("Zipcode").set_index("Zipcode")
    j = z.reindex(zip5.values)
    j.index = ids
    f = pd.DataFrame(index=ids)
    f["zip_missing"] = zip5.isna().astype(int)
    f["zip_unmatched"] = (zip5.notna() & j["Lat"].isna()).astype(int)
    f["zip_in_ca"] = (j["State"] == "CA").astype(int)
    f["geo_lat"], f["geo_lon"] = j["Lat"], j["Long"]
    f["geo_log_pop"] = np.log1p(j["EstimatedPopulation"])
    f["geo_log_wages"] = np.log1p(j["TotalWages"])
    f["geo_wage_per_return"] = j["TotalWages"] / j["TaxReturnsFiled"].replace(0, np.nan)
    for name, (la, lo) in VENUES.items():
        f[f"dist_{name}"] = _haversine(j["Lat"], j["Long"], la, lo)
    f["dist_min"] = f[[f"dist_{k}" for k in VENUES]].min(axis=1)
    return f


# --------------------------------------------------------------------------- #
# Concert (show-type) classification
# --------------------------------------------------------------------------- #
def classify_concerts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per input row with composer-presence, era shares and show-type flags."""
    out = pd.DataFrame(index=df.index)
    what = df["what"].fillna("")
    text = (df["concert.name"].fillna("") + " " + df["who"].fillna("") + " " + what)
    for c in COMPOSERS:
        out[f"comp_{c}"] = what.str.contains(rf"\b{c}", case=False, regex=True).astype(int)
    n_comp = out[[f"comp_{c}" for c in COMPOSERS]].sum(axis=1).replace(0, np.nan)
    for e in ERAS:
        cols = [f"comp_{c}" for c in COMPOSERS if COMPOSER_ERA[c] == e]
        out[f"era_{e}"] = (out[cols].sum(axis=1) / n_comp).fillna(0)
    out["is_vocal"] = text.str.contains(VOCAL_RE).astype(int)
    out["is_holiday"] = text.str.contains(HOLIDAY_RE).astype(int)
    out["is_tentpole"] = text.str.contains(TENTPOLE_RE).astype(int)
    out["n_soloists"] = df["who"].fillna("").apply(
        lambda w: sum(1 for ln in re.split(r"[\r\n]+", w) if ln.strip() and not NON_SOLOIST_RE.search(ln)))
    return out


def _plan_vectors(c15: pd.DataFrame):
    cl = classify_concerts(c15)
    comp = cl[[f"comp_{c}" for c in COMPOSERS]].mean().to_numpy()
    era = cl[[f"era_{e}" for e in ERAS]].mean().to_numpy()
    return comp, era


def _cos(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    num = m @ v
    den = np.linalg.norm(m, axis=1) * np.linalg.norm(v)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


# --------------------------------------------------------------------------- #
# Ticket + taste features
# --------------------------------------------------------------------------- #
def ticket_features(ids: pd.Index, tk: pd.DataFrame, concerts: pd.DataFrame, c15: pd.DataFrame) -> pd.DataFrame:
    t = tk.copy()
    t["start"] = season_start(t["season"])
    t["pl_num"] = pd.to_numeric(t["price.level"], errors="coerce")
    t["pl_special"] = t["price.level"].astype(str).isin(["Adult", "GA", "Youth"])
    t["single"] = (t["no.seats"] == 1).astype(int)
    t["multi"] = (t["multiple.tickets"] == "yes").astype(int)
    t["family"] = (t["location"] == "Family concert").astype(int)
    t["src_known"] = t["marketing.source"].notna().astype(int)
    t["src_web"] = (t["marketing.source"] == "Philharmonia website").astype(int)

    # attach concert info (key = season, set, location); 99% of tickets match
    cc = concerts.copy()
    cl = classify_concerts(cc)
    cc = pd.concat([cc[["season", "set", "location"]], cl], axis=1).drop_duplicates(["season", "set", "location"])
    t = t.merge(cc, on=["season", "set", "location"], how="left")
    taste_cols = ([f"era_{e}" for e in ERAS] + ["is_vocal", "is_holiday", "is_tentpole", "n_soloists"] +
                  [f"comp_{c}" for c in COMPOSERS])
    t["matched"] = t["is_vocal"].notna().astype(int)

    g = t.groupby("account.id")
    f = pd.DataFrame({
        "tk_n_purchases": g.size(),
        "tk_n_tickets": g["no.seats"].sum(),
        "tk_n_seasons": g["season"].nunique(),
        "tk_seasons_since_last": LAST_SEASON - g["start"].max(),
        "tk_n_distinct_concerts": t[t["matched"] == 1].groupby("account.id").apply(
            lambda x: x[["season", "set", "location"]].drop_duplicates().shape[0], include_groups=False),
        "tk_avg_price_level": g["pl_num"].mean(),
        "tk_avg_seats": g["no.seats"].mean(),
        "tk_single_share": g["single"].mean(),
        "tk_any_multiple": g["multi"].max(),
        "tk_family_share": g["family"].mean(),
        "tk_special_price": g["pl_special"].max().astype(int),
        "tk_src_known": g["src_known"].max(),
        "tk_src_web": g["src_web"].max(),
        "tk_matched_share": g["matched"].mean(),
    })
    mode_loc = g["location"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "None")
    taste = t[t["matched"] == 1].groupby("account.id")[taste_cols].mean()

    f = f.reindex(ids)
    has = f["tk_n_purchases"].notna()
    f.insert(0, "tk_has_tickets", has.astype(int))
    f = f.fillna({"tk_n_purchases": 0, "tk_n_tickets": 0, "tk_n_seasons": 0, "tk_seasons_since_last": NO_HISTORY,
                  "tk_n_distinct_concerts": 0, "tk_avg_price_level": -1, "tk_avg_seats": 0,
                  "tk_single_share": 0, "tk_any_multiple": 0, "tk_family_share": 0, "tk_special_price": 0,
                  "tk_src_known": 0, "tk_src_web": 0, "tk_matched_share": 0})
    f["tk_single_multi_show"] = ((f["tk_single_share"] >= 0.5) & (f["tk_n_distinct_concerts"] >= 3)).astype(int)
    loc = mode_loc.reindex(ids).fillna("None")
    f = pd.concat([f, _onehot(loc, TK_LOCATIONS, "tk_loc")], axis=1)

    # taste profile (mean over concerts attended) -> only for the key columns, plus similarity to 2014-15 plan
    tz = taste.reindex(ids).fillna(0)
    keep = ([f"era_{e}" for e in ERAS] + ["is_vocal", "is_holiday", "is_tentpole", "n_soloists"] +
            [f"comp_{c}" for c in KEY_COMPOSERS])
    for c in keep:
        f[f"taste_{c}"] = tz[c]
    plan_comp, plan_era = _plan_vectors(c15)
    f["taste_sim_composer"] = _cos(tz[[f"comp_{c}" for c in COMPOSERS]].to_numpy(), plan_comp)
    f["taste_sim_era"] = _cos(tz[[f"era_{e}" for e in ERAS]].to_numpy(), plan_era)

    # most recent season only (2013-14): "bought tickets to several shows last season" is a natural
    # step toward subscribing, which helps the never-subscribed (cold-start) group
    t13 = t[t["start"] == LAST_SEASON].copy()
    t13["show"] = t13["set"].astype(str) + "|" + t13["location"].astype(str)
    g13 = t13.groupby("account.id")
    r = pd.DataFrame({"tk13_n_tickets": g13["no.seats"].sum(),
                      "tk13_n_shows": g13["show"].nunique()}).reindex(ids).fillna(0)
    f["tk13_n_tickets"] = r["tk13_n_tickets"]
    f["tk13_n_shows"] = r["tk13_n_shows"]
    f["tk13_ge3_shows"] = (r["tk13_n_shows"] >= 3).astype(int)
    return f


# --------------------------------------------------------------------------- #
# Account-ID order (Salesforce-style IDs increase with creation time)
# --------------------------------------------------------------------------- #
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _b62(s: str) -> int:
    n = 0
    for ch in s[6:]:                    # first 6 chars are a constant prefix
        n = n * 62 + _B62.index(ch)
    return n


def id_rank_feature(ids: pd.Index, acc: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank of the account ID among ALL accounts in account.csv (unsupervised, no label)."""
    all_ids = acc["account.id"]
    rank = pd.Series(all_ids.map(_b62).values, index=all_ids.values).rank(pct=True)
    return pd.DataFrame({"acct_id_rank": rank.reindex(ids).values}, index=ids)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_features(ids, tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    T = tables or load_tables()
    ids = pd.Index(pd.Series(ids).values, name="account.id")
    parts = [
        subscription_features(ids, T["subscriptions"]),
        donation_features(ids, T["account"]),
        geo_features(ids, T["account"], T["zipcodes"]),
        ticket_features(ids, T["tickets"], T["concerts"], T["concerts_1415"]),
        id_rank_feature(ids, T["account"]),
    ]
    X = pd.concat(parts, axis=1)
    X = X.astype({c: float for c in X.columns})
    return X
