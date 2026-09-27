"""
Synthetic transaction generator, version 2.

How it works
------------
1. PLAN (no transactions yet): customers and their segments, households,
   devices, networks, trips and device upgrades; the merchant catalogue;
   fraud episodes (archetype, victim, time) and fraud rings (members, shared
   infrastructure, attack wave).
2. LEGITIMATE activity for every customer, including legitimate "anomalies":
   travel, device upgrades, borrowed and household devices, VPN / public
   networks, big and tiny purchases, new categories, shopping bursts.
3. FRAUD transactions for every episode, following its archetype. Only these
   rows get is_fraud=1.
4. LOGIN FAILURES as timestamped events (typos, forgotten passwords,
   credential attacks in an episode's warning period, fraud sessions).
   `failed_logins_24h` of a transaction is then the number of events in the
   24 hours before it: [t - 24h, t).
5. FINALIZE: sort, give every customer strictly increasing timestamps,
   assign transaction ids in the same order as v1, derive is_new_device /
   is_new_location, precursor flags, fraud stages and contexts.

Determinism: every random draw comes from a numpy Generator keyed by
(seed, domain, entity), e.g. (42, CUSTOMER, 17). Changing one customer's or
one episode's parameters does not reshuffle any other entity's draws. Dates
come from the config (never from the clock).
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema
from .config import (
    DEFAULT_CONFIG, DOMESTIC_CITIES, FOREIGN_CITIES, FOREIGN_CITIES_TRAVEL, FOREIGN_CITIES_V1,
    MERCHANT_CATEGORIES, GeneratorConfig,
)

DAY = 86_400
HOUR = 3_600
MINUTE = 60

# random-stream domains
_PLAN, _CUSTOMER, _EPISODE, _RING, _LOGINS, _HOUSEHOLD, _SCHEDULE = range(1, 8)


def _rng(cfg: GeneratorConfig, domain: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=(domain, *keys)))


def _uniform(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def _randint(rng, bounds):
    """Inclusive integer range."""
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _loguniform(rng, bounds):
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def _quota(total: int, shares: dict) -> dict:
    """Largest-remainder allocation of `total` items over `shares` (deterministic)."""
    names = list(shares)
    raw = np.array([shares[n] for n in names], dtype=float)
    raw = raw / raw.sum() * total
    base = np.floor(raw).astype(int)
    order = sorted(range(len(names)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: total - base.sum()]:
        base[i] += 1
    return dict(zip(names, base.tolist()))


class _IdRegistry:
    """Neutral, collision-free identifiers (e.g. DEV_7F3A91C2) derived from the
    seed. Every device that is not a customer's original A/B device uses this
    one format, whether it belongs to a legitimate customer, a household or a
    fraudster, so the id string reveals nothing about fraud."""

    def __init__(self, seed: int):
        self.seed = seed
        self.used = set()

    def new(self, prefix: str, kind: str, *key) -> str:
        salt = 0
        while True:
            digest = hashlib.sha256(f"{self.seed}|{kind}|{key}|{salt}".encode()).hexdigest()[:8].upper()
            ident = f"{prefix}_{digest}"
            if ident not in self.used:
                self.used.add(ident)
                return ident
            salt += 1


# ---- plan ------------------------------------------------------------------------

@dataclass
class Customer:
    idx: int
    customer_id: str
    segment: str
    home_city: str
    household_id: int
    hour_weights: np.ndarray
    median_amount: float
    amount_sigma: float
    category_weights: np.ndarray
    weekly_rate: float
    primary_device: str
    secondary_device: str | None
    secondary_use: float
    household_device: str | None
    household_use: float
    upgrade_device: str | None
    upgrade_time: int | None
    borrow: tuple | None                   # (time, device_id, n_txns)
    home_network: str
    carrier_network: str
    vpn_use: float
    trips: list                            # [(start, end, city, "travel_domestic"/"travel_foreign")]

    def primary_at(self, t: int) -> str:
        if self.upgrade_time is not None and t >= self.upgrade_time:
            return self.upgrade_device
        return self.primary_device

    def trip_at(self, t: int):
        for start, end, city, kind in self.trips:
            if start <= t < end:
                return city, kind
        return None


@dataclass
class Episode:
    key: int                               # stable planning key (not the final episode id)
    customer_idx: int
    archetype: str
    ring_key: int                          # -1 = not a ring episode
    first_time: int                        # planned start of the fraud (seconds from start)
    precursor_start: int | None


@dataclass
class Ring:
    key: int
    member_idxs: list
    wave_start: int
    devices: list
    networks: list
    merchants: list                        # [(merchant_id, category)]
    location: str
    categories: list


@dataclass
class Plan:
    cfg: GeneratorConfig
    customers: list
    episodes: list
    rings: list
    merchants: dict                        # category -> (ids, weights)
    city_carriers: dict
    city_public: dict
    proxy_networks: list
    ids: _IdRegistry
    day_of_month: list
    day_of_week: list


def _hour_weights(rng, peaks, width_bounds, floor_bounds):
    shift = int(rng.integers(-1, 2))
    width = rng.uniform(*width_bounds)
    hours = np.arange(24)
    w = np.zeros(24)
    for peak, weight in peaks:
        d = np.abs(hours - (peak + shift) % 24)
        d = np.minimum(d, 24 - d)
        w += weight * np.exp(-0.5 * (d / width) ** 2)
    w /= w.sum()
    floor = rng.uniform(*floor_bounds)
    return (1 - floor) * w + floor / 24


def _place_trip(rng, trips, length_days, city, kind, horizon) -> bool:
    for _ in range(20):
        start = int(rng.integers(3 * DAY, horizon - (length_days + 1) * DAY))
        end = start + length_days * DAY
        if all(end + DAY <= s or start >= e + DAY for s, e, _, _ in trips):
            trips.append((start, end, city, kind))
            trips.sort()
            return True
    return False


def _plan_trips(rng, cfg, seg, home_city, horizon):
    """Domestic trips get their city here. A foreign trip is placed with city
    None; _assign_foreign_destinations picks the cities for all travellers."""
    trips = []
    if rng.random() < seg["p_domestic_trip"]:
        for _ in range(_randint(rng, cfg.domestic_trips)):
            city = str(rng.choice([c for c in DOMESTIC_CITIES if c != home_city]))
            _place_trip(rng, trips, _randint(rng, cfg.domestic_trip_days), city, "travel_domestic", horizon)
    if rng.random() < seg["p_foreign_trip"]:
        _place_trip(rng, trips, _randint(rng, cfg.foreign_trip_days), None, "travel_foreign", horizon)
    return trips


def _assign_foreign_destinations(cfg, customers, horizon):
    """Every foreign city is visited by at least one legitimate traveller, so no
    city name appears only on fraud. If fewer customers travel abroad than
    there are foreign cities, extra travellers are added (business travellers
    first). The first trips cover each city once, in a seeded random order;
    the rest are drawn with the benign/v1 destination split."""
    rng = _rng(cfg, _PLAN, 2)
    has_trip = lambda c: any(k == "travel_foreign" for *_, k in c.trips)  # noqa: E731
    travellers = [c for c in customers if has_trip(c)]
    if cfg.cover_foreign_destinations and len(travellers) < len(FOREIGN_CITIES):
        order = sorted((c for c in customers if not has_trip(c)),
                       key=lambda c: (c.segment != "business_traveller", float(rng.random())))
        for c in order:
            if len(travellers) >= len(FOREIGN_CITIES):
                break
            r = _rng(cfg, _PLAN, 3, c.idx)
            if _place_trip(r, c.trips, _randint(r, cfg.foreign_trip_days), None, "travel_foreign", horizon):
                travellers.append(c)
    travellers.sort(key=lambda c: c.idx)
    visit_order = rng.permutation(len(travellers))
    cover = [str(x) for x in rng.permutation(FOREIGN_CITIES)] if cfg.cover_foreign_destinations else []
    for rank, k in enumerate(visit_order):
        c = travellers[k]
        if rank < len(cover):
            city = cover[rank]
        else:
            pool = FOREIGN_CITIES_TRAVEL if rng.random() < cfg.p_benign_foreign_destination else FOREIGN_CITIES_V1
            city = str(rng.choice(pool))
        c.trips = sorted((s, e, city if kind == "travel_foreign" else town, kind) for s, e, town, kind in c.trips)


def build_plan(cfg: GeneratorConfig = DEFAULT_CONFIG) -> Plan:
    ids = _IdRegistry(cfg.seed)
    horizon = cfg.days * DAY
    prng = _rng(cfg, _PLAN)

    # shared infrastructure: merchants and networks
    merchants, n = {}, 0
    for cat in MERCHANT_CATEGORIES:
        mids = [f"MER_{n + k:05d}" for k in range(cfg.merchants_per_category)]
        n += cfg.merchants_per_category
        w = 1.0 / np.arange(1, cfg.merchants_per_category + 1) ** 1.1
        merchants[cat] = (mids, w / w.sum())
    all_cities = DOMESTIC_CITIES + FOREIGN_CITIES
    city_carriers = {c: [ids.new("NET", "carrier", c, k) for k in range(cfg.carrier_networks_per_city)] for c in all_cities}
    city_public = {c: [ids.new("NET", "public", c, k) for k in range(cfg.public_networks_per_city)] for c in all_cities}
    proxy_networks = [ids.new("NET", "proxy", k) for k in range(cfg.fraud_proxy_networks)]

    # segments by quota, shuffled over customers
    seg_quota = _quota(cfg.n_customers, {s: v["share"] for s, v in cfg.segments.items()})
    seg_list = [s for s, q in seg_quota.items() for _ in range(q)]
    prng.shuffle(seg_list)

    # households: consecutive household-segment customers grouped into 2-3
    hh_members = [i for i, s in enumerate(seg_list) if s == "household"]
    households, pos, hh_rng = [], 0, _rng(cfg, _HOUSEHOLD)
    while pos < len(hh_members):
        size = _randint(hh_rng, cfg.household_size)
        group = hh_members[pos:pos + size]
        if len(group) == 1 and households:
            households[-1].append(group[0])
        else:
            households.append(group)
        pos += size
    household_of = {m: h + 1 for h, g in enumerate(households) for m in g}
    household_info = {}
    for h, group in enumerate(households, start=1):
        r = _rng(cfg, _HOUSEHOLD, h)
        household_info[h] = dict(
            city=str(r.choice(DOMESTIC_CITIES)),
            device=ids.new("DEV", "household", h),
            network=ids.new("NET", "home", "household", h),
        )

    customers = []
    for i in range(cfg.n_customers):
        r = _rng(cfg, _CUSTOMER, i)
        seg_name = seg_list[i]
        seg = cfg.segments[seg_name]
        hh = household_of.get(i, 0)
        home_city = household_info[hh]["city"] if hh else str(r.choice(DOMESTIC_CITIES))
        cat_w = r.dirichlet(np.full(len(MERCHANT_CATEGORIES), 0.6))
        secondary = f"DEV_{i:04d}_B" if r.random() < cfg.p_secondary_device else None
        upgrade_device, upgrade_time = None, None
        if r.random() < cfg.p_device_upgrade:
            upgrade_device = ids.new("DEV", "upgrade", i)
            upgrade_time = int(r.integers(20 * DAY, horizon - 20 * DAY))
        borrow = None
        if r.random() < cfg.p_borrowed_device:
            # half the time the borrowed phone is another customer's own device
            lender = int(r.integers(0, cfg.n_customers))
            if r.random() < 0.5 and lender != i:
                dev = f"DEV_{lender:04d}_A"
            else:
                dev = ids.new("DEV", "borrowed", i)
            borrow = (int(r.integers(10 * DAY, horizon - 2 * DAY)), dev, _randint(r, (1, 3)))
        customers.append(Customer(
            idx=i, customer_id=f"CUST_{i:04d}", segment=seg_name, home_city=home_city, household_id=hh,
            hour_weights=_hour_weights(r, seg["peaks"], cfg.hour_peak_width, cfg.hour_uniform_floor),
            median_amount=_loguniform(r, seg["amount_median"]), amount_sigma=_uniform(r, cfg.amount_sigma),
            category_weights=cat_w, weekly_rate=_uniform(r, seg["weekly"]),
            primary_device=f"DEV_{i:04d}_A", secondary_device=secondary,
            secondary_use=_uniform(r, cfg.secondary_device_use) if secondary else 0.0,
            household_device=household_info[hh]["device"] if hh else None,
            household_use=_uniform(r, cfg.p_household_device_use) if hh else 0.0,
            upgrade_device=upgrade_device, upgrade_time=upgrade_time, borrow=borrow,
            home_network=household_info[hh]["network"] if hh else ids.new("NET", "home", i),
            carrier_network=str(r.choice(city_carriers[home_city])),
            vpn_use=cfg.vpn_use if r.random() < cfg.p_vpn_user else 0.0,
            trips=_plan_trips(r, cfg, seg, home_city, horizon),
        ))

    _assign_foreign_destinations(cfg, customers, horizon)
    episodes, rings = _plan_fraud(cfg, customers, merchants, ids)
    days = pd.date_range(cfg.start_date, periods=cfg.days, freq="D")
    return Plan(cfg, customers, episodes, rings, merchants, city_carriers, city_public, proxy_networks, ids,
                day_of_month=days.day.tolist(), day_of_week=days.dayofweek.tolist())


def _ring_sizes(rng, total, lo, hi, min_rings):
    """Split `total` ring episodes into at least `min_rings` rings of lo..hi members."""
    n_rings = max(min_rings, int(np.ceil(total / ((lo + hi) / 2))))
    n_rings = min(n_rings, total // lo)
    if n_rings < 1 or n_rings * hi < total:
        raise ValueError(f"cannot split {total} ring episodes into rings of {lo}-{hi}")
    sizes = [lo] * n_rings
    for _ in range(total - lo * n_rings):
        open_rings = [k for k in range(n_rings) if sizes[k] < hi]
        sizes[open_rings[int(rng.integers(len(open_rings)))]] += 1
    return sizes


def _plan_fraud(cfg, customers, merchants, ids):
    horizon = cfg.days * DAY
    rng = _rng(cfg, _SCHEDULE)
    n_episodes = int(round(cfg.episodes_per_customer * cfg.n_customers))
    shares = {a: v["share"] for a, v in cfg.archetypes.items()}
    quota = _quota(n_episodes, shares)
    quota["ring"] = max(quota["ring"], cfg.min_rings * cfg.ring_size[0])

    order = rng.permutation(cfg.n_customers).tolist()
    busy = {i: [] for i in range(cfg.n_customers)}     # planned fraud windows per customer

    def free(i, start, end):
        gap = cfg.min_days_between_episodes * DAY
        return all(end + gap <= s or start >= e + gap for s, e in busy[i])

    # rings: distinct members, one short attack wave each
    rings, episodes, key = [], [], 0
    earliest = cfg.first_episode_day * DAY
    sizes = _ring_sizes(rng, quota["ring"], *cfg.ring_size, cfg.min_rings)
    wave_len = int(cfg.ring_wave_max_hours * HOUR)
    for rk, size in enumerate(sizes):
        members = [order.pop(0) for _ in range(size)]
        rr = _rng(cfg, _RING, rk)
        wave_start = int(rr.integers(earliest, horizon - wave_len - DAY))
        cats = [str(c) for c in rr.choice(["electronics", "online_retail", "entertainment", "travel", "fashion"],
                                          size=2, replace=False)]
        ring_merchants = []
        for c in cats[: _randint(rr, (1, 2))]:
            mids, _ = merchants[c]
            ring_merchants.append((mids[int(rr.integers(len(mids) // 2, len(mids)))], c))   # a real, less popular merchant
        location = str(rr.choice(FOREIGN_CITIES)) if rr.random() < 0.5 else str(rr.choice(DOMESTIC_CITIES))
        ring = Ring(
            key=rk, member_idxs=members, wave_start=wave_start,
            devices=[ids.new("DEV", "ring", rk, k) for k in range(_randint(rr, (1, 3)))],
            networks=[ids.new("NET", "ring", rk, k) for k in range(_randint(rr, (1, 2)))],
            merchants=ring_merchants, location=location, categories=cats,
        )
        rings.append(ring)
        # members are hit in a staggered sequence inside the wave
        offsets = np.sort(rr.uniform(0, wave_len - 8 * HOUR, size=size))
        for m, off in zip(members, offsets):
            first = wave_start + int(off)
            pre = _precursor(cfg, rr, "ring", first)
            episodes.append(Episode(key, m, "ring", rk, first, pre))
            busy[m].append((pre if pre is not None else first, first + wave_len))
            key += 1

    # other archetypes: distinct victims first, a share of them victimised twice
    labels = [a for a in cfg.archetypes if a != "ring" for _ in range(quota[a])]
    rng.shuffle(labels)
    n_repeat = int(round(cfg.repeat_victim_share * len(labels) / (1 + cfg.repeat_victim_share)))
    victims = [order.pop(0) for _ in range(len(labels) - n_repeat)]
    victims += victims[:n_repeat]
    for archetype, victim in zip(labels, victims):
        er = _rng(cfg, _EPISODE, key)
        placed = False
        for _ in range(200):
            first = int(er.integers(earliest, horizon - 6 * DAY))
            pre = _precursor(cfg, er, archetype, first)
            if free(victim, pre if pre is not None else first, first + 6 * DAY):
                placed = True
                break
        if not placed:
            raise RuntimeError(f"could not schedule episode {key} for customer {victim}")
        busy[victim].append((pre if pre is not None else first, first + 6 * DAY))
        episodes.append(Episode(key, victim, archetype, -1, first, pre))
        key += 1
    return episodes, rings


def _precursor(cfg, rng, archetype, first):
    if rng.random() < cfg.archetypes[archetype]["p_precursor"]:
        start = first - int(rng.uniform(*cfg.precursor_days) * DAY)
        return max(start, 7 * DAY)
    return None


# ---- legitimate activity ------------------------------------------------------------

def _merchant(plan, rng, category):
    mids, w = plan.merchants[category]
    return mids[int(rng.choice(len(mids), p=w))]


def _legit_transactions(plan: Plan, c: Customer, logins: list) -> list:
    cfg = plan.cfg
    r = _rng(cfg, _CUSTOMER, c.idx, 1)
    rows = []
    public_cities = plan.city_public

    def make(t, contexts, category=None, amount=None, device=None):
        ctx = list(contexts)
        trip = c.trip_at(t)
        location = trip[0] if trip else c.home_city
        if trip:
            ctx.append(trip[1])
        # device
        if device is None:
            primary = c.primary_at(t)
            u = r.random()
            if c.upgrade_time is not None and c.upgrade_time <= t < c.upgrade_time + cfg.upgrade_overlap_days * DAY:
                if u < 0.25:
                    device = c.primary_device                      # old phone still in use for a while
                else:
                    device = primary
                    ctx.append("device_upgrade")
            elif c.household_device and u < c.household_use:
                device = c.household_device
                ctx.append("household_device")
            elif c.secondary_device and u < c.household_use + c.secondary_use:
                device = c.secondary_device
            else:
                device = primary
        # network
        u = r.random()
        if c.vpn_use and u < c.vpn_use:
            network = plan.proxy_networks[int(r.integers(len(plan.proxy_networks)))]
            ctx.append("vpn")
        elif trip:
            pool = plan.city_carriers[location] if u < 0.7 else public_cities[location]
            network = pool[int(r.integers(len(pool)))]
        elif u < cfg.p_public_network:
            network = public_cities[location][int(r.integers(len(public_cities[location])))]
            ctx.append("public_network")
        elif u < cfg.p_public_network + cfg.p_carrier_network:
            network = c.carrier_network
        else:
            network = c.home_network
        # category / amount
        if category is None:
            u = r.random()
            if u < cfg.p_big_purchase:
                category = str(r.choice(["electronics", "travel", "fashion", "online_retail"]))
                amount = c.median_amount * _uniform(r, cfg.big_purchase_multiple)
                ctx.append("big_purchase")
            elif u < cfg.p_big_purchase + cfg.p_small_purchase:
                category = str(r.choice(["grocery", "dining", "fuel", "entertainment"]))
                amount = _uniform(r, cfg.small_purchase_inr)
                ctx.append("small_purchase")
            elif u < cfg.p_big_purchase + cfg.p_small_purchase + cfg.p_new_category:
                rare = np.argsort(c.category_weights)[:4]
                category = MERCHANT_CATEGORIES[int(r.choice(rare))]
                ctx.append("new_category")
            else:
                w = c.category_weights.copy()
                if trip:
                    for boost in ("travel", "dining", "fuel"):
                        w[MERCHANT_CATEGORIES.index(boost)] += 0.15
                    w /= w.sum()
                category = MERCHANT_CATEGORIES[int(r.choice(len(w), p=w))]
        if amount is None:
            amount = c.median_amount * float(np.exp(r.normal(0, c.amount_sigma)))
            if plan.day_of_month[min(t // DAY, cfg.days - 1)] in cfg.salary_days:
                amount *= 1.2
        rows.append(dict(
            t=int(t), customer_idx=c.idx, amount=round(max(float(amount), 1.0), 2), merchant_category=category,
            device_id=device, location=location, merchant_id=_merchant(plan, r, category), network_id=network,
            is_fraud=0, fraud_type="none", episode_key=-1, ring_key=-1, legit_context=ctx,
        ))

    horizon = cfg.days * DAY
    for d in range(cfg.days):
        rate = c.weekly_rate / 7.0
        if plan.day_of_week[d] >= 5:
            rate *= 1.15
        if plan.day_of_month[d] in cfg.salary_days:
            rate *= cfg.salary_rate_uplift
        if c.trip_at(d * DAY + 12 * HOUR):
            rate *= 1.3
        n = int(r.poisson(rate))
        day_times = []
        for _ in range(n):
            hour = int(r.choice(24, p=c.hour_weights))
            t = d * DAY + hour * HOUR + int(r.integers(0, HOUR))
            day_times.append(t)
            make(t, [])
        if day_times and r.random() < cfg.p_burst_day:
            anchor = day_times[int(r.integers(len(day_times)))]
            for _ in range(_randint(r, cfg.burst_extra_txns)):
                t = anchor + int(r.integers(60, HOUR))
                if t < horizon:
                    make(t, ["burst"], category=str(r.choice(["online_retail", "dining", "entertainment"])))

    # a trip always shows up in the data: at least one payment while away
    for t_start, t_end, _, _ in c.trips:
        if not any(t_start <= row["t"] < t_end for row in rows):
            day = t_start // DAY + 1
            make(day * DAY + int(r.choice(24, p=c.hour_weights)) * HOUR + int(r.integers(0, HOUR)), [])

    if c.borrow:
        t0, dev, k = c.borrow
        for j in range(k):
            make(t0 + j * int(r.integers(10 * MINUTE, 2 * HOUR)), ["borrowed_device"], device=dev)

    # legitimate login failures: isolated typos and forgot-password spells right before a transaction
    lr = _rng(cfg, _LOGINS, c.idx)
    for _ in range(int(lr.poisson(cfg.typo_failures_per_day * cfg.days))):
        logins.append((c.idx, int(lr.integers(0, horizon)), "typo"))
    n_forgot = int(lr.poisson(cfg.forgot_password_per_day * cfg.days))
    for _ in range(min(n_forgot, len(rows))):
        anchor = rows[int(lr.integers(len(rows)))]["t"]
        t = anchor - int(lr.integers(5 * MINUTE, HOUR))
        for j in range(_randint(lr, cfg.forgot_password_failures)):
            logins.append((c.idx, t + 20 * j, "forgot_password"))
    return rows


# ---- fraud ------------------------------------------------------------------------

def _pick_hour(rng, c: Customer, off_hours: bool) -> int:
    w = c.hour_weights
    if off_hours:
        inv = w.max() - w + 1e-3
        return int(rng.choice(24, p=inv / inv.sum()))
    return int(rng.choice(24, p=w))


def _fraud_transactions(plan: Plan, ep: Episode, logins: list) -> list:
    cfg = plan.cfg
    a = cfg.archetypes[ep.archetype]
    c = plan.customers[ep.customer_idx]
    r = _rng(cfg, _EPISODE, ep.key, 1)
    ring = plan.rings[ep.ring_key] if ep.ring_key >= 0 else None
    fraud_device = plan.ids.new("DEV", "fraud", ep.key)

    # ---- per-episode choices
    if ring is None:
        uses_new_device = r.random() < a["p_new_device"]
        u = r.random()
        if u < a["p_foreign"]:
            location = str(r.choice(FOREIGN_CITIES))
        elif u < a["p_foreign"] + a["p_other_city"]:
            location = str(r.choice([x for x in DOMESTIC_CITIES if x != c.home_city]))
        else:
            location = c.home_city
        if r.random() < a["p_fraud_network"]:
            network = plan.proxy_networks[int(r.integers(len(plan.proxy_networks)))]
        else:
            network = c.home_network if r.random() < 0.5 else c.carrier_network

    # ---- times
    if ring is not None:
        first = ep.first_time            # the ring's coordinated wave decides the time
    else:
        session_off = r.random() < a["p_off_hours"]
        first = (ep.first_time // DAY) * DAY + _pick_hour(r, c, session_off) * HOUR + int(r.integers(0, HOUR))
    kinds = []                                              # (time, kind) kind in {"txn", "probe", "cashout"}
    if ep.archetype == "card_testing_cashout":
        n_probes = _randint(r, a["n_probes"])
        span = _uniform(r, a["probe_span_hours"]) * HOUR
        times = first + np.sort(r.uniform(0, span, n_probes)).astype(int)
        times[0] = first
        kinds += [(int(t), "probe") for t in times]
        if r.random() < a["p_cashout"]:
            last = int(times[-1])
            for _ in range(_randint(r, a["n_cashout"])):
                kinds.append((last + int(_uniform(r, a["cashout_delay_hours"]) * HOUR), "cashout"))
    else:
        n = _randint(r, a["n_txns"])
        span = _uniform(r, a["span_hours"]) * HOUR
        offsets = np.sort(r.uniform(0, span, n)).astype(int)
        offsets[0] = 0
        long_episode = a["span_hours"][1] > 24
        for off in offsets:
            t = first + int(off)
            if long_episode and off > 0:
                # multi-day episodes: each later transaction picks its own hour
                t = (t // DAY) * DAY + _pick_hour(r, c, r.random() < a["p_off_hours"]) * HOUR + int(r.integers(0, HOUR))
                t = max(t, first + 60)
            kinds.append((t, "txn"))

    # ---- login failures around the fraud session
    if r.random() < a["p_attack_logins"]:
        for _ in range(_randint(r, a["attack_failures"])):
            logins.append((c.idx, first - int(r.integers(MINUTE, 2 * HOUR)), "fraud_session"))
    if ep.precursor_start is not None:
        lo, hi = ep.precursor_start, max(ep.precursor_start + HOUR, first - HOUR)
        for _ in range(_randint(r, cfg.precursor_attack_bursts)):
            burst = int(r.integers(lo, hi))
            for j in range(_randint(r, cfg.precursor_failures_per_burst)):
                logins.append((c.idx, burst + 30 * j, "credential_attack"))

    rows = []
    if ring is not None:
        fresh_device = plan.ids.new("DEV", "ring_member", ep.key)
    for t, kind in kinds:
        # device / network / location / merchant
        if ring is not None:
            device = ring.devices[int(r.integers(len(ring.devices)))] if r.random() < a["p_ring_device"] else fresh_device
            network = (ring.networks[int(r.integers(len(ring.networks)))] if r.random() < a["p_ring_network"]
                       else plan.proxy_networks[int(r.integers(len(plan.proxy_networks)))])
            location = ring.location if r.random() < a["p_ring_location"] else c.home_city
        else:
            device = fraud_device if uses_new_device else c.primary_at(t)
        merchant = None
        if kind == "probe":
            category = str(r.choice(a["shift_categories"]))
            amount = _uniform(r, a["probe_amount_inr"])
        else:
            if ring is not None and r.random() < a["p_ring_merchant"]:
                merchant, category = ring.merchants[int(r.integers(len(ring.merchants)))]
            elif r.random() < a["p_category_shift"]:
                pool = ring.categories if ring is not None else a["shift_categories"]
                category = str(r.choice(pool))
            else:
                w = c.category_weights
                category = MERCHANT_CATEGORIES[int(r.choice(len(w), p=w))]
            amount = c.median_amount * _loguniform(r, a["amount"]) * float(np.exp(r.normal(0, 0.1)))
        rows.append(dict(
            t=int(t), customer_idx=c.idx, amount=round(max(float(amount), 1.0), 2), merchant_category=category,
            device_id=device, location=location, merchant_id=merchant or _merchant(plan, r, category),
            network_id=network, is_fraud=1, fraud_type=ep.archetype, episode_key=ep.key,
            ring_key=ep.ring_key, legit_context=[],
        ))
    return rows


def count_preceding(event_times: np.ndarray, txn_times: np.ndarray, window: int = DAY) -> np.ndarray:
    """For each transaction time t, the number of events e with t - window <= e < t.
    `event_times` must be sorted. An event at exactly t is not counted (it is
    not "preceding"); an event exactly 24h earlier is."""
    event_times = np.asarray(event_times)
    txn_times = np.asarray(txn_times)
    return np.searchsorted(event_times, txn_times, "left") - np.searchsorted(event_times, txn_times - window, "left")


# ---- assemble ----------------------------------------------------------------------

@dataclass
class GeneratedData:
    transactions: pd.DataFrame           # schema.RAW_COLUMNS
    customers: pd.DataFrame              # schema.CUSTOMER_COLUMNS
    episodes: pd.DataFrame               # schema.EPISODE_COLUMNS
    login_failures: pd.DataFrame         # schema.LOGIN_FAILURE_COLUMNS
    config: GeneratorConfig


def generate(cfg: GeneratorConfig = DEFAULT_CONFIG) -> GeneratedData:
    plan = build_plan(cfg)
    logins, rows = [], []
    for c in plan.customers:
        rows += _legit_transactions(plan, c, logins)
    for ep in plan.episodes:
        rows += _fraud_transactions(plan, ep, logins)
    return _finalize(plan, rows, logins)


def _finalize(plan: Plan, rows: list, logins: list) -> GeneratedData:
    cfg = plan.cfg
    start = pd.Timestamp(cfg.start_date)
    horizon = cfg.days * DAY
    df = pd.DataFrame(rows)
    df = df[(df["t"] >= 0) & (df["t"] < horizon)]
    # stable order; within the same second legitimate rows before fraud, then planning order
    df = df.sort_values(["customer_idx", "t", "is_fraud"], kind="mergesort").reset_index(drop=True)

    # strictly increasing timestamps per customer (ties are pushed forward by a second)
    t = df["t"].to_numpy().copy()
    cust = df["customer_idx"].to_numpy()
    for k in range(1, len(t)):
        if cust[k] == cust[k - 1] and t[k] <= t[k - 1]:
            t[k] = t[k - 1] + 1
    df["t"] = t

    # login-failure events -> failed_logins_24h = events in [t - 24h, t)
    lg = pd.DataFrame(logins, columns=["customer_idx", "t", "source"])
    lg = lg[(lg["t"] >= 0) & (lg["t"] < horizon)].sort_values(["customer_idx", "t", "source"], kind="mergesort")
    failed = np.zeros(len(df), dtype=int)
    forgot = np.zeros(len(df), dtype=bool)
    ev_by_c = {k: g for k, g in lg.groupby("customer_idx")}
    for k, idx in df.groupby("customer_idx").indices.items():
        ev = ev_by_c.get(k)
        if ev is None:
            continue
        et = ev["t"].to_numpy()
        tt = t[idx]
        failed[idx] = count_preceding(et, tt)
        ft = np.sort(ev.loc[ev["source"] == "forgot_password", "t"].to_numpy())
        if len(ft):
            forgot[idx] = count_preceding(ft, tt) > 0
    df["failed_logins_24h"] = failed

    # episodes: final ids ordered by (first fraud time, customer), fraud stages, precursor flags
    fr = df[df["is_fraud"] == 1]
    firsts = fr.groupby("episode_key")["t"].min()
    lasts = fr.groupby("episode_key")["t"].max()
    counts = fr.groupby("episode_key").size()
    eps = {e.key: e for e in plan.episodes}
    order = sorted(firsts.index, key=lambda k: (firsts[k], eps[k].customer_idx))
    episode_id = {k: i + 1 for i, k in enumerate(order)}
    ring_order = sorted(plan.rings, key=lambda rg: (rg.wave_start, rg.key))
    ring_id = {rg.key: i + 1 for i, rg in enumerate(ring_order)}

    df["fraud_episode_id"] = df["episode_key"].map(episode_id).fillna(0).astype(int)
    df["fraud_ring_id"] = df["ring_key"].map(ring_id).fillna(0).astype(int)
    is_first = (df["is_fraud"] == 1) & (df["t"] == df["episode_key"].map(firsts))
    df["fraud_stage"] = np.where(df["is_fraud"] == 1, np.where(is_first, "first", "subsequent"), "none")

    precursor = np.zeros(len(df), dtype=int)
    ep_rows = []
    for k in order:
        e = eps[k]
        pre_start = e.precursor_start
        if pre_start is not None:
            mask = ((df["customer_idx"] == e.customer_idx) & (df["is_fraud"] == 0)
                    & (df["t"] >= pre_start) & (df["t"] < firsts[k])).to_numpy()
            precursor[mask] = 1
        ep_rows.append(dict(
            fraud_episode_id=episode_id[k], customer_id=f"CUST_{e.customer_idx:04d}", fraud_type=e.archetype,
            fraud_ring_id=ring_id.get(e.ring_key, 0),
            precursor_start=str(start + pd.Timedelta(seconds=pre_start)) if pre_start is not None else "",
            first_fraud_time=str(start + pd.Timedelta(seconds=int(firsts[k]))),
            last_fraud_time=str(start + pd.Timedelta(seconds=int(lasts[k]))),
            n_fraud_transactions=int(counts[k]),
        ))
    df["is_precursor"] = precursor

    ctx = df["legit_context"].tolist()
    for k in np.flatnonzero(forgot & (df["is_fraud"].to_numpy() == 0)):
        ctx[k] = ctx[k] + ["forgot_password"]
    df["legit_context"] = ["|".join(sorted(set(x))) if x else "none" for x in ctx]

    df["customer_id"] = [f"CUST_{k:04d}" for k in df["customer_idx"]]
    df["timestamp"] = (start + pd.to_timedelta(df["t"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["transaction_id"] = [f"TXN_{k:07d}" for k in range(len(df))]
    df["is_new_device"] = (~df.duplicated(["customer_id", "device_id"])).astype(int)
    df["is_new_location"] = (~df.duplicated(["customer_id", "location"])).astype(int)
    seg = {c.idx: c.segment for c in plan.customers}
    df["customer_segment"] = df["customer_idx"].map(seg)
    transactions = df[schema.RAW_COLUMNS].reset_index(drop=True)

    customers = pd.DataFrame([dict(
        customer_id=c.customer_id, customer_segment=c.segment, home_city=c.home_city, household_id=c.household_id,
        primary_device=c.primary_device, secondary_device=c.secondary_device or "",
        upgrade_device=c.upgrade_device or "",
        upgrade_date=str(start + pd.Timedelta(seconds=c.upgrade_time)) if c.upgrade_time is not None else "",
        has_foreign_trip=int(any(k == "travel_foreign" for *_, k in c.trips)),
        n_domestic_trips=sum(k == "travel_domestic" for *_, k in c.trips),
    ) for c in plan.customers], columns=schema.CUSTOMER_COLUMNS)

    episodes = pd.DataFrame(ep_rows, columns=schema.EPISODE_COLUMNS)
    login_failures = pd.DataFrame({
        "customer_id": [f"CUST_{k:04d}" for k in lg["customer_idx"]],
        "timestamp": (start + pd.to_timedelta(lg["t"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "source": lg["source"].to_numpy(),
    }, columns=schema.LOGIN_FAILURE_COLUMNS)
    return GeneratedData(transactions, customers, episodes, login_failures, cfg)
