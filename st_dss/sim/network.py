"""Road network and demand generation for SUMO."""
import os
import subprocess
import sys

import sumolib


def sumo_home():
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    import sumo  # pip package eclipse-sumo
    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    return sumo.SUMO_HOME


def binary(name):
    sumo_home()
    return sumolib.checkBinary(name)


def build_network(cfg, out_dir, log=print):
    s = cfg["sim"]
    if s.get("city"):
        from .osm_city import city_network
        return city_network(cfg, s["city"], out_dir, log)
    if s.get("net_file"):
        from ..config import root_path
        return root_path(s["net_file"])
    net = os.path.join(out_dir, "grid.net.xml")
    if not os.path.exists(net):
        subprocess.run([binary("netgenerate"), "--grid",
                        "--grid.number", str(s["grid_number"]),
                        "--grid.length", str(s["grid_length"]),
                        "--default.lanenumber", str(s["lanes"]),
                        "--default.speed", "13.89",
                        "--tls.guess", "true",
                        "--no-turnarounds", "true",
                        "-o", net], check=True, capture_output=True)
    return net


# Indian urban traffic mix (two-wheelers dominate). Used when sim.vehicle_mix = india.
VTYPES_INDIA = """<additional>
  <vTypeDistribution id="india_mix">
    <vType id="two_wheeler" vClass="motorcycle" length="2.0" width="0.8" minGap="1.0" maxSpeed="16.7" accel="3.0" decel="5.0" sigma="0.6" probability="0.55"/>
    <vType id="car" vClass="passenger" length="4.3" minGap="2.0" maxSpeed="16.7" sigma="0.5" probability="0.28"/>
    <vType id="auto_rickshaw" vClass="taxi" length="2.8" width="1.3" minGap="1.2" maxSpeed="12.5" accel="1.5" sigma="0.6" probability="0.12"/>
    <vType id="bus" vClass="bus" length="11.0" minGap="2.5" maxSpeed="13.9" accel="1.0" decel="3.5" probability="0.05"/>
  </vTypeDistribution>
</additional>
"""


def vtypes_file(cfg, out_dir):
    if cfg["sim"].get("vehicle_mix") != "india":
        return None
    p = os.path.join(out_dir, "vtypes.add.xml")
    with open(p, "w") as f:
        f.write(VTYPES_INDIA)
    return p


def build_trips(cfg, net, out_dir, seed):
    s = cfg["sim"]
    trips = os.path.join(out_dir, f"trips_seed{seed}.rou.xml")
    if not os.path.exists(trips):
        script = os.path.join(sumo_home(), "tools", "randomTrips.py")
        vt = vtypes_file(cfg, out_dir)
        attrs = 'departLane="best" departSpeed="max"' + (' type="india_mix"' if vt else "")
        cmd = [sys.executable, script, "-n", net, "-o", trips,
               "-b", "0", "-e", str(s["demand_end"]), "-p", str(s["demand_period"]),
               "--seed", str(seed), "--fringe-factor", "5",
               "--min-distance", str(s.get("min_trip_distance", 600)),
               "--prefix", "v", "--validate", "--trip-attributes", attrs]
        if vt:
            cmd += ["--additional-files", vt]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
        if r.returncode != 0:
            raise RuntimeError("randomTrips failed:\n" + r.stderr[-2000:])
    return trips


def _boundary(net, margin=0.1):
    """Node is 'on the boundary' if it lies within `margin` x extent of the network
    bounding box (works for the regular grid and for irregular OSM networks)."""
    xs = [n.getCoord()[0] for n in net.getNodes()]
    ys = [n.getCoord()[1] for n in net.getNodes()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return lambda n: (n.getCoord()[0] <= x0 + mx or n.getCoord()[0] >= x1 - mx
                      or n.getCoord()[1] <= y0 + my or n.getCoord()[1] >= y1 - my)


def boundary_edges(net):
    """(sources, sinks): edges leaving / entering the network boundary."""
    on_b = _boundary(net)
    src = [e.getID() for e in net.getEdges() if on_b(e.getFromNode()) and not on_b(e.getToNode())]
    snk = [e.getID() for e in net.getEdges() if on_b(e.getToNode()) and not on_b(e.getFromNode())]
    return src, snk


def interior_edges(net_path):
    """Edges whose both end nodes are not on the network boundary."""
    net = sumolib.net.readNet(net_path)
    on_b = _boundary(net)
    return [e.getID() for e in net.getEdges()
            if not on_b(e.getFromNode()) and not on_b(e.getToNode())
            and e.allows("passenger")]


def street_name(net, edge_id):
    name = net.getEdge(edge_id).getName()
    return f"{name} ({edge_id})" if name else edge_id
