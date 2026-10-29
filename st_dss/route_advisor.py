"""Route advisor: recommended route between two road segments using the current forecast.

Edge cost (same as the simulation):
  travel_time : length / forecast_speed
  recovery    : travel_time x (1 + alpha (1 - forecast_speed / free_speed))
Dijkstra over the SUMO edge graph (respects allowed turns).
"""
import heapq


def edge_cost(edge, ratio, mode="recovery", alpha=1.0):
    v_free = edge.getSpeed()
    r = min(max(ratio, 0.02), 1.0)
    tt = edge.getLength() / max(r * v_free, 0.5)
    return tt * (1 + alpha * (1 - r)) if mode == "recovery" else tt


def best_route(net, ratios, src, dst, mode="recovery", alpha=1.0):
    """ratios: {edge_id: forecast speed / free speed} (missing -> 1.0 = free flow).
    Returns (list of edge ids, estimated travel time in s) or (None, inf)."""
    start, goal = net.getEdge(src), net.getEdge(dst)
    dist = {src: edge_cost(start, ratios.get(src, 1.0), mode, alpha)}
    prev, pq = {}, [(dist[src], src)]
    while pq:
        d, eid = heapq.heappop(pq)
        if eid == dst:
            break
        if d > dist.get(eid, float("inf")):
            continue
        for nxt in net.getEdge(eid).getOutgoing():
            nid = nxt.getID()
            nd = d + edge_cost(nxt, ratios.get(nid, 1.0), mode, alpha)
            if nd < dist.get(nid, float("inf")):
                dist[nid], prev[nid] = nd, eid
                heapq.heappush(pq, (nd, nid))
    if dst not in dist:
        return None, float("inf")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    eta = sum(edge_cost(net.getEdge(e), ratios.get(e, 1.0), "travel_time") for e in path)
    return path, eta


def directions(net, path, ratios=None):
    """Group a route into named road segments:
    [{'road', 'distance_m', 'time_min', 'edges'}]. Unnamed links are merged into the
    neighbouring segment so the list reads like turn-by-turn directions."""
    ratios = ratios or {}
    out = []
    for eid in path:
        e = net.getEdge(eid)
        name = e.getName() or ""
        t = e.getLength() / max(min(ratios.get(eid, 1.0), 1.0) * e.getSpeed(), 0.5)
        if out and (name == "" or name == out[-1]["road"]):
            out[-1]["distance_m"] += e.getLength()
            out[-1]["time_min"] += t / 60
            out[-1]["edges"].append(eid)
        else:
            out.append(dict(road=name or "unnamed street", distance_m=e.getLength(),
                            time_min=t / 60, edges=[eid]))
    return out


def roads_of(steps):
    return [s["road"] for s in steps if s["road"] != "unnamed street"]


def lonlat_path(net, eid):
    """Edge shape as [[lon, lat], ...] (None if the network has no geo-projection)."""
    try:
        return [list(net.convertXY2LonLat(x, y)) for x, y in net.getEdge(eid).getShape()]
    except Exception:
        return None
