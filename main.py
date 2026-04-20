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
    b:        float = 0.0

class YBusRequest(BaseModel):
    branches: List[Branch]

class ZBusBranch(BaseModel):
    type:     int
    from_bus: int
    to_bus:   int
    r:        float
    x:        float

class ZBusRequest(BaseModel):
    branches: List[ZBusBranch]

class BusData(BaseModel):
    bus:   int
    type:  str      # "slack", "pq", "pv"
    P:     float = 0.0
    Q:     float = 0.0
    Vmag:  float = 1.0   # specified |V| for slack and PV buses

class LoadFlowRequest(BaseModel):
    branches:  List[Branch]
    buses:     List[BusData]
    max_iter:  int   = 100
    tolerance: float = 1e-4

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

        t  = b.type
        k  = b.from_bus
        p  = b.to_bus

        if t == 1:
            # Ref → New bus
            n += 1
            Z_new = np.zeros((n, n), dtype=complex)
            if n > 1:
                Z_new[:n-1, :n-1] = Z
            Z_new[n-1, n-1] = zb
            Z = Z_new
            steps.append(f"Type 1: Ref → New bus {p}. Z({p},{p}) = {fmt(zb)}")

        elif t == 2:
            ki = k - 1
            if ki < 0 or ki >= n:
                return JSONResponse({"error": f"Branch {idx+1}: bus {k} does not exist yet."}, status_code=400)
            n += 1
            Z_new = np.zeros((n, n), dtype=complex)
            Z_new[:n-1, :n-1] = Z
            for i in range(n-1):
                Z_new[n-1, i] = Z[ki, i]
                Z_new[i, n-1] = Z[i, ki]
            Z_new[n-1, n-1] = Z[ki, ki] + zb
            Z = Z_new
            steps.append(f"Type 2: Bus {k} → New bus {p}. Z({p},{p}) = {fmt(Z[n-1,n-1])}")

        elif t == 3:
            # For Type 3: from_bus = 0 (reference), to_bus = existing bus k
            ki = p - 1   # use to_bus as the existing bus
            if ki < 0 or ki >= n:
                return JSONResponse({"error": f"Branch {idx+1}: bus {p} does not exist yet. Add it first with Type 1 or Type 2."}, status_code=400)
            na = n + 1
            Za = np.zeros((na, na), dtype=complex)
            Za[:n, :n] = Z
            for i in range(n):
                Za[n, i] = -Z[ki, i]
                Za[i, n] = -Z[i, ki]
            Za[n, n] = Z[ki, ki] + zb
            Z = Za[:n, :n] - np.outer(Za[:n, n], Za[n, :n]) / Za[n, n]
            steps.append(f"Type 3: Ref → Existing bus {p} (link). Kron reduction applied.")

        elif t == 4:
            ji = k - 1
            ki = p - 1
            if ji < 0 or ji >= n:
                return JSONResponse({"error": f"Branch {idx+1}: bus {k} does not exist."}, status_code=400)
            if ki < 0 or ki >= n:
                return JSONResponse({"error": f"Branch {idx+1}: bus {p} does not exist."}, status_code=400)
            na = n + 1
            Za = np.zeros((na, na), dtype=complex)
            Za[:n, :n] = Z
            for i in range(n):
                Za[n, i] = Z[ji, i] - Z[ki, i]
                Za[i, n] = Z[i, ji] - Z[i, ki]
            Za[n, n] = Z[ji, ji] - 2*Z[ji, ki] + Z[ki, ki] + zb
            Z = Za[:n, :n] - np.outer(Za[:n, n], Za[n, :n]) / Za[n, n]
            steps.append(f"Type 4: Bus {k} → Bus {p} (link). Kron reduction applied.")

        else:
            return JSONResponse({"error": f"Branch {idx+1}: invalid type {t}."}, status_code=400)

    return {"n_buses": n, "zbus": matrix_to_list(Z), "steps": steps}

# ─────────────────────────────────────────────────────────────
#  LOAD FLOW — GAUSS-SEIDEL (Hadi Saadat method)
# ─────────────────────────────────────────────────────────────

@app.post("/api/loadflow")
async def compute_loadflow(data: LoadFlowRequest):
    """
    Gauss-Seidel Load Flow — exactly as in Hadi Saadat, Power System Analysis.

    Sign convention (per Saadat):
      - Slack bus : V specified (magnitude + angle=0). Bus 1 typically.
      - PQ bus    : P and Q are SCHEDULED NET INJECTIONS in pu.
                    Load bus  → P negative, Q negative  (e.g. P=-2.566, Q=-1.102)
                    Generator → P positive (net P = Pgen - Pload)
      - PV bus    : P scheduled (pu), |V| specified. Q computed each iteration.

    Flat start: all non-slack buses initialised to 1.0∠0°.
    Acceleration factor: 1.6 (Saadat default).
    """
    branches = data.branches
    bus_data = data.buses

    if not branches:
        return JSONResponse({"error": "No branch data provided."}, status_code=400)
    if not bus_data:
        return JSONResponse({"error": "No bus data provided."}, status_code=400)

    n = max(max(b.from_bus, b.to_bus) for b in branches)

    # ── Build Y-bus ───────────────────────────────────────────
    Y = np.zeros((n, n), dtype=complex)
    for b in branches:
        i, j = b.from_bus - 1, b.to_bus - 1
        z = complex(b.r, b.x)
        if abs(z) < 1e-12:
            continue
        y_s = 1.0 / z
        y_sh = complex(0, b.b / 2.0)
        Y[i][i] += y_s + y_sh
        Y[j][j] += y_s + y_sh
        Y[i][j] -= y_s
        Y[j][i] -= y_s

    # ── Bus setup ─────────────────────────────────────────────
    # Flat start: V = 1 + j0 for all buses
    V        = [complex(1.0, 0.0)] * n
    P        = [0.0] * n
    Q        = [0.0] * n
    Vmag_sp  = [1.0] * n       # specified |V| for slack/PV
    btype    = ["pq"] * n
    slack    = -1

    for bd in bus_data:
        idx = bd.bus - 1
        if idx < 0 or idx >= n:
            return JSONResponse(
                {"error": f"Bus {bd.bus} out of range (system has {n} buses)."},
                status_code=400)
        t = bd.type.lower().strip()
        btype[idx] = t
        if t == "slack":
            slack       = idx
            Vmag_sp[idx] = bd.Vmag
            V[idx]      = complex(bd.Vmag, 0.0)   # slack voltage fixed
        elif t == "pv":
            P[idx]       = bd.P
            Vmag_sp[idx] = bd.Vmag
        else:   # pq
            P[idx] = bd.P
            Q[idx] = bd.Q

    if slack == -1:
        return JSONResponse({"error": "No Slack bus defined."}, status_code=400)

    # ── Gauss-Seidel iterations ───────────────────────────────
    # Equation (Saadat 6.28):
    #
    #   V_i^(k+1) = (1 / Y_ii) *
    #               [ (P_i - j*Q_i) / conj(V_i^(k))
    #                 - sum_{j≠i} Y_ij * V_j^(latest) ]
    #
    # "latest" = already-updated value for j already processed this iteration.
    # After update, apply acceleration:  V_new = V_old + α*(V_calc - V_old)
    # For PV bus: after updating V, fix |V| to Vmag_sp while keeping angle.

    alpha     = 1.6       # acceleration factor (Saadat uses 1.6)
    tol       = data.tolerance
    converged = False
    iters     = 0

    for iteration in range(data.max_iter):
        V_prev = V.copy()

        for i in range(n):
            if btype[i] == "slack":
                continue          # slack voltage never changes

            if btype[i] == "pv":
                # Compute Q from current voltages (Saadat eq 6.34)
                I_i  = sum(Y[i][j] * V[j] for j in range(n))
                S_i  = V[i] * I_i.conjugate()
                Q[i] = S_i.imag   # computed reactive power

            # Saadat eq 6.28
            sigma = sum(Y[i][j] * V[j] for j in range(n) if j != i)
            V_calc = (1.0 / Y[i][i]) * (
                complex(P[i], -Q[i]) / V[i].conjugate() - sigma
            )

            # Acceleration
            V_acc = V_prev[i] + alpha * (V_calc - V_prev[i])

            if btype[i] == "pv":
                # Fix magnitude, keep new angle
                ang  = cmath.phase(V_acc)
                V[i] = Vmag_sp[i] * complex(cmath.cos(ang), cmath.sin(ang))
            else:
                V[i] = V_acc

        # Convergence check — max change in voltage
        max_dV = max(abs(V[i] - V_prev[i]) for i in range(n) if btype[i] != "slack")
        iters  = iteration + 1
        if max_dV < tol:
            converged = True
            break

    # ── Compute slack bus P and Q from final voltages ─────────
    I_slack = sum(Y[slack][j] * V[j] for j in range(n))
    S_slack = V[slack] * I_slack.conjugate()
    P[slack] = S_slack.real
    Q[slack] = S_slack.imag

    # ── Build results ─────────────────────────────────────────
    results = []
    for i in range(n):
        results.append({
            "bus":        i + 1,
            "type":       btype[i].upper(),
            "voltage_pu": round(abs(V[i]), 6),
            "angle_deg":  round(cmath.phase(V[i]) * 180 / cmath.pi, 6),
            "P_pu":       round(P[i], 6),
            "Q_pu":       round(Q[i], 6),
            "V_rect":     f"{round(V[i].real,6)} {'+' if V[i].imag>=0 else '-'} j{round(abs(V[i].imag),6)}"
        })

    return {
        "iterations": iters,
        "converged":  converged,
        "tolerance":  tol,
        "buses":      results
    }
