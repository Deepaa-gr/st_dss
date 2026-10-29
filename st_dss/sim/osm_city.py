"""Real road networks for Western Tamil Nadu cities from OpenStreetMap.

    python run_pipeline.py --stage sim --city erode

1. Downloads the drivable roads of a ~2.6 km x 2.6 km box around the city centre
   from the Overpass API (needs internet on YOUR machine; cached in data/osm/).
   If the download fails, export the same box manually from openstreetmap.org
   (Export -> Manually select a different area) and save it as data/osm/<city>.osm
2. Converts it to a SUMO network with netconvert (left-hand traffic for India,
   signal guessing, junction joining, street names kept).
"""
import os
import subprocess
import time
import urllib.parse
import urllib.request

from .network import binary

# city centre (lat, lon) — main bus stand / central junction
CITIES = {
    "coimbatore": dict(lat=11.0168, lon=76.9558, label="Coimbatore (Gandhipuram)"),
    "erode":      dict(lat=11.3410, lon=77.7172, label="Erode (Central / Bus Stand)"),
    "tiruppur":   dict(lat=11.1085, lon=77.3411, label="Tiruppur (Town centre)"),
    "salem":      dict(lat=11.6643, lon=78.1460, label="Salem (New Bus Stand / Five Roads)"),
    "namakkal":   dict(lat=11.2189, lon=78.1674, label="Namakkal (Town centre)"),
}

ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
              "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street")
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def bbox(city, half_km=1.3):
    c = CITIES[city]
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * 0.98)          # cos(11 deg) ~ 0.98
    return (c["lat"] - dlat, c["lon"] - dlon, c["lat"] + dlat, c["lon"] + dlon)  # s, w, n, e


def download_osm(city, path, half_km=1.3, log=print):
    s, w, n, e = bbox(city, half_km)
    q = (f'[out:xml][timeout:180];(way["highway"~"^({ROAD_TYPES})$"]({s},{w},{n},{e}););'
         f'(._;>;);out body;')
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for url in OVERPASS:
        for attempt in range(2):
            try:
                log(f"[osm] downloading {city} roads from {url.split('/')[2]} ...")
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "ST-DSS research"})
                with urllib.request.urlopen(req, timeout=240) as r, open(path, "wb") as f:
                    f.write(r.read())
                if os.path.getsize(path) > 2000:
                    log(f"[osm] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
                    return path
                last = "empty response"
            except Exception as ex:  # network errors, 429, timeouts
                last = ex
                time.sleep(5)
    raise RuntimeError(
        f"Could not download OSM data for {city} ({last}).\n"
        f"Manual fix: open https://www.openstreetmap.org/export, select the box "
        f"S={s:.4f} W={w:.4f} N={n:.4f} E={e:.4f}, click Export and save it as {path}")


def convert(osm_path, net_path, log=print, speed_factor=1.0):
    cmd = [binary("netconvert"), "--osm-files", osm_path, "-o", net_path,
           "--speed.factor", str(speed_factor),             # OSM defaults (primary 100 km/h) -> Indian city speeds
           "--lefthand", "true",                                 # India: drive on the left
           "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
           "--junctions.join", "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
           "--edges.join", "--remove-edges.isolated",
           "--keep-edges.by-vclass", "passenger",
           "--output.street-names", "--output.original-names",
           "--no-turnarounds.except-deadend", "true",
           "--no-warnings"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("netconvert failed:\n" + r.stderr[-2000:])
    log(f"[osm] SUMO network written: {net_path}")
    return net_path


def city_network(cfg, city, out_dir, log=print):
    from ..config import root_path
    if city not in CITIES:
        raise ValueError(f"unknown city '{city}'. Choose from: {', '.join(CITIES)}")
    osm_dir = root_path(os.path.join("data", "osm"))
    os.makedirs(osm_dir, exist_ok=True)
    osm = os.path.join(osm_dir, f"{city}.osm")
    sf = float(cfg["sim"].get("speed_factor", 1.0))
    suffix = "" if sf == 1.0 else f"_speed{sf:g}"
    net = os.path.join(out_dir, f"{city}{suffix}.net.xml")
    if os.path.exists(net):          # already converted earlier: no download needed
        return net
    if not os.path.exists(osm):
        download_osm(city, osm, cfg["sim"].get("city_half_km", 1.3), log)
    convert(osm, net, log, sf)
    return net
