from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import numpy as np
import cmath

app = FastAPI(title="SmartGrid Cloud Deployment")
templates = Jinja2Templates(directory="templates")

# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

class Branch(BaseModel):
    from_bus: int
    to_bus:   int
    r:        float
    x:        float
    b:        float = 0.0   # line charging susceptance (optional)

class YBusRequest(BaseModel):
    branches: List[Branch]

class ZBusBranch(BaseModel):
    type:     int   # 1,2,3,4
    from_bus: int   # 0 = reference bus
    to_bus:   int
    r:        float
    x:        float

class ZBusRequest(BaseModel):
    branches: List[ZBusBranch]

class BusData(BaseModel):
    bus:  int
    type: str       # "slack" or "pq"
    P:    float = 0.0
    Q:    float = 0.0

class LoadFlowRequest(BaseModel):
    branches:  List[Branch]
    buses:     List[BusData]
    max_iter:  int   = 100
    tolerance: float = 1e-6

# ─────────────────────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/ybus", response_class=HTMLResponse)
async def ybus_page(request: Request):
    return templates.TemplateResponse(request=request, name="ybus.html")

@app.get("/zbus", response_class=HTMLResponse)
async def zbus_page(request: Request):
    return templates.TemplateResponse(request=request, name="zbus.html")

@app.get("/loadflow", response_class=HTMLResponse)
async def loadflow_page(request: Request):
    return templates.TemplateResponse(request=request, name="loadflow.html")

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def fmt(c: complex) -> str:
    r = round(c.real, 5)
    i = round(c.imag, 5)
    sign = "+" if i >= 0 else "-"
    return f"{r} {sign} j{abs(i)}"

def matrix_to_list(M: np.ndarray) -> list:
    n = M.shape[0]
    return [[fmt(M[r][c]) for c in range(n)] for r in range(n)]

# ─────────────────────────────────────────────────────────────
#  Y-BUS API
# ─────────────────────────────────────────────────────────────

@app.post("/api/ybus")
async def compute_ybus(data: YBusRequest):
    branches = data.branches
    if not branches:
        return JSONResponse({"error": "No branch data provided."}, status_code=400)

    n = max(max(b.from_bus, b.to_bus) for b in branches)
    Y = np.zeros((n, n), dtype=complex)

    for b in branches:
        i = b.from_bus - 1
        j = b.to_bus - 1
        z = complex(b.r, b.x)
        if abs(z) < 1e-12:
            return JSONResponse(
                {"error": f"Zero impedance on branch {b.from_bus}-{b.to_bus}."},
                status_code=400)
        y_series = 1.0 / z
        y_shunt  = complex(0, b.b / 2.0)
        Y[i][i] += y_series + y_shunt
        Y[j][j] += y_series + y_shunt
        Y[i][j] -= y_series
        Y[j][i] -= y_series

    return {"n_buses": n, "ybus": matrix_to_list(Y)}

# ─────────────────────────────────────────────────────────────
#  Z-BUS BUILDING ALGORITHM API
# ─────────────────────────────────────────────────────────────

@app.post("/api/zbus")
async def compute_zbus(data: ZBusRequest):
    """
    Z-Bus Building Algorithm — 4 Modification Types:
      Type 1: Reference bus → New bus        (tree branch, adds new bus)
      Type 2: Existing bus k → New bus p     (tree branch, adds new bus)
      Type 3: Reference bus → Existing bus k (link, no new bus, Kron reduction)
      Type 4: Existing bus j → Existing bus k(link, no new bus, Kron reduction)
    """
    branches = data.branches
    if not branches:
        return JSONResponse({"error": "No branch data provided."}, status_code=400)

    Z = np.zeros((0, 0), dtype=complex)
    n = 0
    steps = []

    for idx, b in enumerate(branches):
        zb = complex(b.r, b.x)
        if abs(zb) < 1e-12:
            return JSONResponse(
                {"error": f"Zero impedance on branch {idx+1}."}, status_code=400)

        t    = b.type
        k    = b.from_bus   # existing bus (1-indexed); 0 = reference
        p    = b.to_bus     # new or existing bus (1-indexed)

        if t == 1:
            # ── Type 1: Ref → New bus p ───────────────────────────────
            # New row/col added; Z_pp = zb; all cross terms = 0
            n += 1
            Z_new = np.zeros((n, n), dtype=complex)
            if n > 1:
                Z_new[:n-1, :n-1] = Z
            Z_new[n-1, n-1] = zb
            Z = Z_new
            steps.append(f"Type 1: Added new bus {p} from reference. Z({p},{p}) = {fmt(zb)}")

        elif t == 2:
            # ── Type 2: Existing bus k → New bus p ───────────────────
            # Z_pi = Z_ip = Z_ki for all existing i
            # Z_pp = Z_kk + zb
            ki = k - 1
            if ki < 0 or ki >= n:
                return JSONResponse(
                    {"error": f"Branch {idx+1}: from_bus {k} does not exist yet."},
                    status_code=400)
            n += 1
            Z_new = np.zeros((n, n), dtype=complex)
            Z_new[:n-1, :n-1] = Z
            for i in range(n-1):
                Z_new[n-1, i] = Z[ki, i]
                Z_new[i, n-1] = Z[i, ki]
            Z_new[n-1, n-1] = Z[ki, ki] + zb
            Z = Z_new
            steps.append(f"Type 2: Added new bus {p} from bus {k}. Z({p},{p}) = {fmt(Z[n-1,n-1])}")

        elif t == 3:
            # ── Type 3: Ref → Existing bus k (link) ──────────────────
            # Augment with extra row l:
            #   Z_li = -Z_ki,  Z_ll = Z_kk + zb
            # Then Kron-reduce out row/col l
            ki = k - 1
            if ki < 0 or ki >= n:
                return JSONResponse(
                    {"error": f"Branch {idx+1}: bus {k} does not exist."},
                    status_code=400)
            na = n + 1
            Za = np.zeros((na, na), dtype=complex)
            Za[:n, :n] = Z
            for i in range(n):
                Za[n, i] = -Z[ki, i]
                Za[i, n] = -Z[i, ki]
            Za[n, n] = Z[ki, ki] + zb
            # Kron reduction: eliminate row/col n
            Z = Za[:n, :n] - np.outer(Za[:n, n], Za[n, :n]) / Za[n, n]
            steps.append(f"Type 3: Link from reference to bus {k}. Kron reduction applied.")

        elif t == 4:
            # ── Type 4: Existing bus j → Existing bus k (link) ───────
            # Augment with extra row l:
            #   Z_li = Z_ji - Z_ki,  Z_ll = Z_jj - 2*Z_jk + Z_kk + zb
            # Then Kron-reduce out row/col l
            ji = k - 1       # from_bus is j
            ki = p - 1       # to_bus is k
            if ji < 0 or ji >= n:
                return JSONResponse(
                    {"error": f"Branch {idx+1}: bus {k} does not exist."},
                    status_code=400)
            if ki < 0 or ki >= n:
                return JSONResponse(
                    {"error": f"Branch {idx+1}: bus {p} does not exist."},
                    status_code=400)
            na = n + 1
            Za = np.zeros((na, na), dtype=complex)
            Za[:n, :n] = Z
            for i in range(n):
                Za[n, i] = Z[ji, i] - Z[ki, i]
                Za[i, n] = Z[i, ji] - Z[i, ki]
            Za[n, n] = Z[ji, ji] - 2*Z[ji, ki] + Z[ki, ki] + zb
            # Kron reduction
            Z = Za[:n, :n] - np.outer(Za[:n, n], Za[n, :n]) / Za[n, n]
            steps.append(f"Type 4: Link from bus {k} to bus {p}. Kron reduction applied.")

        else:
            return JSONResponse(
                {"error": f"Branch {idx+1}: invalid type {t}. Must be 1, 2, 3 or 4."},
                status_code=400)

    return {
        "n_buses": n,
        "zbus":    matrix_to_list(Z),
        "steps":   steps
    }

# ─────────────────────────────────────────────────────────────
#  LOAD FLOW — GAUSS-SEIDEL API
# ─────────────────────────────────────────────────────────────

@app.post("/api/loadflow")
async def compute_loadflow(data: LoadFlowRequest):
    branches = data.branches
    bus_data = data.buses

    if not branches or not bus_data:
        return JSONResponse({"error": "Branch or bus data missing."}, status_code=400)

    n = max(max(b.from_bus, b.to_bus) for b in branches)

    # Build Y-bus
    Y = np.zeros((n, n), dtype=complex)
    for b in branches:
        i, j = b.from_bus - 1, b.to_bus - 1
        z = complex(b.r, b.x)
        if abs(z) < 1e-12:
            continue
        y_series = 1.0 / z
        y_shunt  = complex(0, b.b / 2.0)
        Y[i][i] += y_series + y_shunt
        Y[j][j] += y_series + y_shunt
        Y[i][j] -= y_series
        Y[j][i] -= y_series

    # Initialize
    V         = [complex(1, 0)] * n
    P         = [0.0] * n
    Q         = [0.0] * n
    slack_bus = 0

    for bd in bus_data:
        idx = bd.bus - 1
        if bd.type == "slack":
            slack_bus = idx
        else:
            P[idx] = bd.P
            Q[idx] = bd.Q

    # Gauss-Seidel iterations
    converged        = False
    iterations_done  = 0

    for iteration in range(data.max_iter):
        V_old = V.copy()
        for i in range(n):
            if i == slack_bus:
                continue
            sigma = sum(Y[i][j] * V[j] for j in range(n) if j != i)
            S_i   = complex(P[i], -Q[i])
            V[i]  = (1 / Y[i][i]) * (S_i / V[i].conjugate() - sigma)
        max_change      = max(abs(V[i] - V_old[i]) for i in range(n))
        iterations_done = iteration + 1
        if max_change < data.tolerance:
            converged = True
            break

    results = []
    for i in range(n):
        results.append({
            "bus":        i + 1,
            "type":       "Slack" if i == slack_bus else "PQ",
            "voltage_pu": round(abs(V[i]), 6),
            "angle_deg":  round(cmath.phase(V[i]) * 180 / cmath.pi, 6),
            "P_pu":       P[i],
            "Q_pu":       Q[i]
        })

    return {
        "iterations": iterations_done,
        "converged":  converged,
        "buses":      results
    }
