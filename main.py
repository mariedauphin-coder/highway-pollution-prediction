"""
=============================================================
  POLLUTION PREDICTION ON HIGHWAY
  Complete System — Single File for VS Code
  
  Run:  python main.py
  
  Requirements:
    pip install ultralytics opencv-python deep-sort-realtime
                easyocr fastapi uvicorn torch torchvision
                pyyaml numpy Pillow
=============================================================
"""

import cv2
import json
import time
import re
import sqlite3
import random
import threading
import argparse
import numpy as np
import torch

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "weights":         "yolov8n.pt",
    "conf":            0.40,
    "iou":             0.45,
    "camera_id":       "HWY-A-CAM-003",
    "highway":         "Gyeongbu Expressway",
    "window_min":      10,
    "fps_target":      10,
    "dashboard_port":  8000,
    "vehicle_classes": {2:"sedan", 5:"bus", 7:"truck", 3:"motorcycle"},
    "class_colors": {
        "sedan":      (255, 180,  50),
        "bus":        ( 30, 180, 220),
        "truck":      ( 50,  50, 220),
        "motorcycle": (200,  80, 255),
        "suv":        (  0, 200, 255),
        "pickup":     (255,  80, 180),
        "van":        ( 80, 220, 100),
        "unknown":    (180, 180, 180),
    },
}

EMISSION_FACTORS = {
    ("sedan",  "Fuel"):  (142, 0.85),
    ("sedan",  "HeV"):   ( 98, 0.40),
    ("sedan",  "BEV"):   (  0, 0.00),
    ("sedan",  "FCEV"):  (  0, 0.00),
    ("suv",    "Fuel"):  (192, 1.10),
    ("suv",    "HeV"):   (135, 0.55),
    ("suv",    "BEV"):   (  0, 0.00),
    ("truck",  "Fuel"):  (310, 0.60),
    ("truck",  "BEV"):   (  0, 0.00),
    ("pickup", "Fuel"):  (255, 1.20),
    ("van",    "Fuel"):  (220, 0.70),
    ("van",    "BEV"):   (  0, 0.00),
    ("bus",    "Fuel"):  (620, 1.80),
    ("bus",    "BEV"):   (  0, 0.00),
    ("bus",    "FCEV"):  (  0, 0.00),
}

VEHICLE_DB = [
    ("Hyundai","Ioniq 5",   2021,None,"BEV",  0),
    ("Hyundai","Ioniq 6",   2022,None,"BEV",  0),
    ("Hyundai","Nexo",      2018,None,"FCEV", 0),
    ("Hyundai","Tucson",    2021,None,"HeV",132),
    ("Hyundai","Tucson",    2015,2020,"Fuel",158),
    ("Hyundai","Santa Fe",  2021,None,"HeV",148),
    ("Hyundai","Sonata",    2020,None,"HeV",110),
    ("Hyundai","Kona Electric",2018,None,"BEV",0),
    ("Hyundai","Grandeur",  2020,None,"HeV",142),
    ("Kia",    "EV6",       2021,None,"BEV",  0),
    ("Kia",    "EV9",       2023,None,"BEV",  0),
    ("Kia",    "Niro EV",   2018,None,"BEV",  0),
    ("Kia",    "Niro",      2016,None,"HeV",104),
    ("Kia",    "Sportage",  2022,None,"HeV",136),
    ("Kia",    "Sorento",   2021,None,"HeV",149),
    ("Kia",    "K5",        2020,None,"HeV",114),
    ("Genesis","GV60",      2022,None,"BEV",  0),
    ("Genesis","GV70 Electric",2022,None,"BEV",0),
    ("Tesla",  "Model 3",   2019,None,"BEV",  0),
    ("Tesla",  "Model Y",   2021,None,"BEV",  0),
    ("Tesla",  "Model S",   2014,None,"BEV",  0),
    ("Toyota", "Prius",     2016,None,"HeV", 94),
    ("Toyota", "RAV4 Hybrid",2019,None,"HeV",118),
    ("Toyota", "Mirai",     2015,None,"FCEV", 0),
    ("BMW",    "i4",        2022,None,"BEV",  0),
    ("BMW",    "330e",      2019,None,"HeV", 39),
    ("Mercedes","EQS",      2022,None,"BEV",  0),
    ("Hyundai","Xcient FCEV",2020,None,"FCEV",0),
    ("Hyundai","Electric Bus",2019,None,"BEV",0),
    ("Hyundai","Universe",  2010,None,"Fuel",620),
]

HIGHWAY_SPEED_KMH   = 95.0
WINDOW_DURATION_H   = 10.0 / 60.0
DIST_PER_VEHICLE_KM = HIGHWAY_SPEED_KMH * WINDOW_DURATION_H

BLUE_HSV_LOWER  = np.array([100, 80,  50])
BLUE_HSV_UPPER  = np.array([130, 255, 255])
GREEN_HSV_LOWER = np.array([ 40,  60,  50])
GREEN_HSV_UPPER = np.array([ 85, 255, 255])


# ══════════════════════════════════════════════════════════════
# MODULE 3 — PLATE DETECTION
# ══════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    plate_text:  Optional[str]
    plate_color: str
    engine_type: str
    confidence:  float
    color_ratio: float


class PlateDetector:
    def find_plate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        h, w = vehicle_crop.shape[:2]
        search = vehicle_crop[int(h * 0.55):, :]
        hsv    = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        w_mask = cv2.inRange(hsv, np.array([0,   0, 180]), np.array([180,  40, 255]))
        y_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([ 35, 255, 255]))
        combined = cv2.bitwise_or(w_mask, y_mask)
        kernel   = np.ones((3, 15), np.uint8)
        cleaned  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 40 or ch < 10:
                continue
            aspect = cw / max(ch, 1)
            if 3.0 < aspect < 8.0:
                score = cw * ch
                if score > best_score:
                    best_score = score
                    ry   = y + int(h * 0.55)
                    best = vehicle_crop[ry:ry+ch, x:x+cw]
        return best

    def detect_plate_color(self, plate_img: np.ndarray) -> Tuple[str, float]:
        if plate_img is None or plate_img.size == 0:
            return "standard", 0.0
        hsv   = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        total = plate_img.shape[0] * plate_img.shape[1]
        blue_r   = np.sum(cv2.inRange(hsv, BLUE_HSV_LOWER,  BLUE_HSV_UPPER)  > 0) / total
        green_r  = np.sum(cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER) > 0) / total
        yellow_r = np.sum(cv2.inRange(hsv, np.array([20,100,100]),
                                           np.array([35,255,255])) > 0) / total
        if blue_r   > 0.05: return "blue",     blue_r
        if green_r  > 0.05: return "green",    green_r
        if yellow_r > 0.15: return "yellow",   yellow_r
        return "standard", 0.0

    def color_to_engine(self, color: str) -> str:
        return {"blue":"BEV","green":"FCEV","yellow":"Fuel","standard":"Fuel"}.get(color,"Fuel")

    def analyze_vehicle(self, vehicle_crop: np.ndarray) -> PlateResult:
        plate_img        = self.find_plate_region(vehicle_crop)
        color, ratio     = self.detect_plate_color(plate_img)
        return PlateResult(
            plate_text  = None,
            plate_color = color,
            engine_type = self.color_to_engine(color),
            confidence  = 0.0,
            color_ratio = ratio,
        )


# ══════════════════════════════════════════════════════════════
# MODULE 4 — ENGINE IDENTIFICATION
# ══════════════════════════════════════════════════════════════

@dataclass
class EngineResult:
    engine_type:  str
    source:       str
    confidence:   float
    co2_g_per_km: float
    co_g_per_km:  float


class EngineIdentifier:
    def __init__(self):
        self.db = VEHICLE_DB

    def get_factors(self, vtype: str, engine: str) -> Tuple[float, float]:
        return EMISSION_FACTORS.get(
            (vtype.lower(), engine),
            EMISSION_FACTORS.get(("sedan","Fuel"), (142, 0.85))
        )

    def lookup_brand_model(self, brand: str, model: str) -> Optional[dict]:
        best, best_score = None, 0
        for row in self.db:
            db_brand, db_model, yf, yt, engine, co2 = row
            bs = SequenceMatcher(None, brand.lower(), db_brand.lower()).ratio()
            ms = SequenceMatcher(None, model.lower(), db_model.lower()).ratio()
            if bs < 0.7 or ms < 0.6:
                continue
            score = (bs + ms) / 2
            if score > best_score:
                best_score = score
                best = {"brand":db_brand,"model":db_model,
                        "engine_type":engine,"co2":co2,"score":score}
        return best

    def identify(self, plate_color: str, plate_text: Optional[str] = None,
                 vehicle_type: str = "sedan", brand: Optional[str] = None,
                 model: Optional[str] = None, color_ratio: float = 0.0) -> EngineResult:
        # Path A: plate color
        if plate_color == "blue":
            co2, co = self.get_factors(vehicle_type, "BEV")
            return EngineResult("BEV","plate_color",
                                min(0.95, 0.7+color_ratio*5), co2, co)
        if plate_color == "green":
            co2, co = self.get_factors(vehicle_type, "FCEV")
            return EngineResult("FCEV","plate_color",
                                min(0.95, 0.7+color_ratio*5), co2, co)
        # Path B: DB lookup
        if brand and model:
            match = self.lookup_brand_model(brand, model)
            if match:
                eng = match["engine_type"]
                co2, co = self.get_factors(vehicle_type, eng)
                return EngineResult(eng,"db_lookup",match["score"],co2,co)
        # Fallback
        co2, co = self.get_factors(vehicle_type, "Fuel")
        return EngineResult("Fuel","default",0.5,co2,co)


# ══════════════════════════════════════════════════════════════
# MODULE 5 — EMISSION ESTIMATION
# ══════════════════════════════════════════════════════════════

@dataclass
class WindowReport:
    camera_id:       str
    window_start:    str
    window_end:      str
    total_vehicles:  int   = 0
    total_co2_g:     float = 0.0
    total_co_g:      float = 0.0
    total_co2_kg:    float = 0.0
    by_engine:       Dict  = field(default_factory=dict)
    by_type:         Dict  = field(default_factory=dict)
    electric_pct:    float = 0.0

    def to_dict(self):
        return asdict(self)

    def print_report(self):
        print(f"\n{'='*55}")
        print(f"  EMISSION REPORT  —  {self.camera_id}")
        print(f"  {self.window_start}  →  {self.window_end}")
        print(f"{'='*55}")
        print(f"  Vehicles      : {self.total_vehicles}")
        print(f"  CO2           : {self.total_co2_kg:.3f} kg")
        print(f"  CO            : {self.total_co_g:.1f} g")
        print(f"  Zero-emission : {self.electric_pct:.1f}%")
        print(f"\n  By engine:")
        for eng, data in sorted(self.by_engine.items()):
            bar = "█" * min(20, int(data["co2_g"]/max(self.total_co2_g,1)*20))
            print(f"    {eng:6s}: {data['count']:3d} veh | "
                  f"CO2={data['co2_g']:>8.0f}g  {bar}")
        print(f"{'='*55}")


class EmissionEstimator:
    def __init__(self, camera_id: str = "HWY-A-CAM-003",
                 window_minutes: float = 10.0):
        self.camera_id      = camera_id
        self.window_minutes = window_minutes
        self.window_start   = datetime.now()
        self.counts         = defaultdict(int)
        self.history: List[WindowReport] = []
        print(f"[Emission] Camera:{camera_id} | "
              f"Window:{window_minutes}min | "
              f"Dist/vehicle:{DIST_PER_VEHICLE_KM:.1f}km")

    def add_vehicle(self, vehicle_type: str, engine_type: str, count: int = 1):
        self.counts[(vehicle_type.lower(), engine_type)] += count

    def close_window(self) -> WindowReport:
        now = datetime.now()
        by_engine = defaultdict(lambda:{"count":0,"co2_g":0,"co_g":0})
        by_type   = defaultdict(lambda:{"count":0,"co2_g":0,"co_g":0})
        total_co2 = 0; total_co = 0; total_veh = 0

        for (vtype, engine), count in self.counts.items():
            co2_km, co_km = EMISSION_FACTORS.get(
                (vtype,engine), EMISSION_FACTORS.get(("sedan","Fuel"),(142,0.85)))
            co2 = co2_km * DIST_PER_VEHICLE_KM * count
            co  = co_km  * DIST_PER_VEHICLE_KM * count
            total_co2 += co2; total_co += co; total_veh += count
            by_engine[engine]["count"]  += count
            by_engine[engine]["co2_g"]  += co2
            by_engine[engine]["co_g"]   += co
            by_type[vtype]["count"]     += count
            by_type[vtype]["co2_g"]     += co2

        zero_em = sum(v["count"] for k,v in by_engine.items()
                      if k in ("BEV","FCEV"))
        report = WindowReport(
            camera_id      = self.camera_id,
            window_start   = self.window_start.isoformat(timespec="seconds"),
            window_end     = now.isoformat(timespec="seconds"),
            total_vehicles = total_veh,
            total_co2_g    = round(total_co2, 2),
            total_co_g     = round(total_co,  2),
            total_co2_kg   = round(total_co2/1000, 3),
            by_engine      = {k:dict(v) for k,v in by_engine.items()},
            by_type        = {k:dict(v) for k,v in by_type.items()},
            electric_pct   = round(zero_em/max(total_veh,1)*100, 1),
        )
        self.history.append(report)
        self.counts       = defaultdict(int)
        self.window_start = now
        return report

    def get_summary(self) -> dict:
        if not self.history:
            return {}
        return {
            "windows":          len(self.history),
            "total_vehicles":   sum(r.total_vehicles for r in self.history),
            "total_co2_kg":     round(sum(r.total_co2_kg for r in self.history), 3),
            "total_co_g":       round(sum(r.total_co_g  for r in self.history), 1),
            "avg_electric_pct": round(sum(r.electric_pct for r in self.history)
                                      / len(self.history), 1),
        }


# ══════════════════════════════════════════════════════════════
# MODULE 6 — DASHBOARD
# ══════════════════════════════════════════════════════════════

dashboard_state = {
    "window_num":    0,
    "total_co2_kg":  0.0,
    "total_co_g":    0.0,
    "total_vehicles":0,
    "windows":       [],
    "current":       {},
    "camera_id":     CONFIG["camera_id"],
    "highway":       CONFIG["highway"],
}

def state_next_window():
    s = dashboard_state
    s["window_num"] += 1
    rush = 1.5 if s["window_num"] % 6 in [2,3] else 1.0
    DIST = DIST_PER_VEHICLE_KM
    counts = {
        ("sedan","Fuel"):int(random.randint(22,32)*rush),
        ("sedan","HeV"): int(random.randint(8, 14)*rush),
        ("sedan","BEV"): int(random.randint(5, 12)*rush),
        ("suv",  "Fuel"):int(random.randint(14,22)*rush),
        ("suv",  "HeV"): int(random.randint(5, 10)*rush),
        ("suv",  "BEV"): int(random.randint(2,  6)*rush),
        ("truck","Fuel"):int(random.randint(4,  9)*rush),
        ("pickup","Fuel"):int(random.randint(2, 5)*rush),
        ("van",  "Fuel"):int(random.randint(3,  7)*rush),
        ("bus",  "Fuel"):int(random.randint(1,  4)),
        ("bus",  "BEV"): int(random.randint(0,  2)),
        ("sedan","FCEV"):int(random.randint(0,  2)),
    }
    co2_g=0; co_g=0; n_veh=0; by_type={}; by_engine={}
    for (vt,eng),cnt in counts.items():
        c2,co = EMISSION_FACTORS.get((vt,eng),(142,0.85))
        co2_g+=c2*DIST*cnt; co_g+=co*DIST*cnt; n_veh+=cnt
        by_type[vt]  = by_type.get(vt,0)+cnt
        by_engine[eng]= by_engine.get(eng,0)+cnt
    zero = sum(c for (v,e),c in counts.items() if e in ("BEV","FCEV"))
    window = {
        "window":      s["window_num"],
        "time":        datetime.now().strftime("%H:%M"),
        "vehicles":    n_veh,
        "co2_kg":      round(co2_g/1000,2),
        "co_g":        round(co_g,1),
        "electric_pct":round(zero/max(n_veh,1)*100,1),
        "by_type":     by_type,
        "by_engine":   by_engine,
    }
    s["windows"].append(window)
    if len(s["windows"])>12: s["windows"]=s["windows"][-12:]
    s["total_co2_kg"]   += window["co2_kg"]
    s["total_co_g"]     += window["co_g"]
    s["total_vehicles"] += n_veh
    s["current"]         = window
    return window

# Seed dashboard with 6 windows
for _ in range(6):
    state_next_window()

DASHBOARD_HTML = open("dashboard.html").read() if Path("dashboard.html").exists() else ""

app = FastAPI(title="Highway Pollution Dashboard")
app.add_middleware(CORSMiddleware,allow_origins=["*"],
                   allow_methods=["*"],allow_headers=["*"])

@app.get("/api/live")
def api_live(): return JSONResponse(dashboard_state["current"])

@app.get("/api/summary")
def api_summary():
    return JSONResponse({
        "total_vehicles":  dashboard_state["total_vehicles"],
        "total_co2_kg":    round(dashboard_state["total_co2_kg"],2),
        "total_co_g":      round(dashboard_state["total_co_g"],1),
        "windows_recorded":dashboard_state["window_num"],
        "camera_id":       dashboard_state["camera_id"],
        "highway":         dashboard_state["highway"],
    })

@app.get("/api/windows")
def api_windows(): return JSONResponse(dashboard_state["windows"])

@app.get("/api/tick")
def api_tick(): return JSONResponse(state_next_window())

@app.get("/", response_class=HTMLResponse)
def api_dashboard():
    if DASHBOARD_HTML:
        return HTMLResponse(DASHBOARD_HTML)
    return HTMLResponse("<h2>Put dashboard.html in the same folder as main.py</h2>")


# ══════════════════════════════════════════════════════════════
# MASTER PIPELINE — All modules connected
# ══════════════════════════════════════════════════════════════

class HighwayPollutionPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n" + "="*55)
        print("  HIGHWAY POLLUTION PREDICTION SYSTEM")
        print("="*55)
        print(f"  Device  : {self.device.upper()}")
        print(f"  Camera  : {CONFIG['camera_id']}")
        print("="*55)

        print("\n[1/5] Loading YOLOv8 detector...")
        self.detector = YOLO(CONFIG["weights"])

        print("[2/5] Initializing DeepSORT tracker...")
        self.tracker = DeepSort(max_age=30, n_init=3,
                                max_iou_distance=0.7,
                                max_cosine_distance=0.3,
                                nn_budget=100)

        print("[3/5] Loading plate detector...")
        self.plate = PlateDetector()

        print("[4/5] Loading engine identifier...")
        self.engine_id = EngineIdentifier()

        print("[5/5] Initializing emission estimator...")
        self.estimator = EmissionEstimator(
            camera_id      = CONFIG["camera_id"],
            window_minutes = CONFIG["window_min"],
        )

        self.frame_count    = 0
        self.tracked        = {}
        self.total_counted  = 0
        self.window_stats   = defaultdict(int)
        self.window_start_t = time.time()

        print("\n[OK] All modules loaded!\n")

    def process_frame(self, frame: np.ndarray, timestamp: float):
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Module 1: Detect
        results = self.detector(frame, conf=CONFIG["conf"],
                                device=self.device, verbose=False,
                                classes=list(CONFIG["vehicle_classes"].keys()))
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CONFIG["vehicle_classes"]: continue
                conf = float(box.conf[0])
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                vtype = CONFIG["vehicle_classes"][cls_id]
                detections.append(([x1,y1,x2-x1,y2-y1], conf, vtype))

        # Module 2: Track
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active = []

        for track in tracks:
            if not track.is_confirmed(): continue
            tid   = track.track_id
            vtype = track.det_class or "sedan"
            x1,y1,x2,y2 = [int(v) for v in track.to_ltrb()]
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)

            if tid not in self.tracked:
                # Module 3: Plate
                crop      = frame[y1:y2, x1:x2] if x2>x1 and y2>y1 else frame
                plate_res = self.plate.analyze_vehicle(crop)
                # Module 4: Engine
                eng_res   = self.engine_id.identify(
                    plate_color  = plate_res.plate_color,
                    vehicle_type = vtype,
                    color_ratio  = plate_res.color_ratio,
                )
                self.tracked[tid] = {
                    "type":        vtype,
                    "engine":      eng_res.engine_type,
                    "co2_per_km":  eng_res.co2_g_per_km,
                    "plate_color": plate_res.plate_color,
                    "frame_count": 0,
                    "positions":   [],
                    "counted":     False,
                }

            v = self.tracked[tid]
            v["frame_count"] += 1
            v["positions"].append(((x1+x2)//2, (y1+y2)//2))

            # Module 5: Count once confirmed
            if v["frame_count"] >= 3 and not v["counted"]:
                v["counted"] = True
                self.total_counted += 1
                # Register with emission estimator
                self.estimator.add_vehicle(v["type"], v["engine"])
                self.window_stats[v["engine"]] += 1
                print(f"  [+] #{tid:3d} {v['type']:8s} {v['engine']:5s} "
                      f"CO2={v['co2_per_km']:>5.0f}g/km "
                      f"plate={v['plate_color']}")

            active.append((tid, v, (x1,y1,x2,y2)))

        # Close window if time elapsed
        elapsed_min = (time.time() - self.window_start_t) / 60
        if elapsed_min >= CONFIG["window_min"]:
            report = self.estimator.close_window()
            report.print_report()
            self.window_stats   = defaultdict(int)
            self.window_start_t = time.time()

        return self._draw(frame.copy(), active, timestamp), active

    def _draw(self, frame, active, timestamp):
        h, w = frame.shape[:2]
        dot_bgr = {"BEV":(80,175,76),"FCEV":(165,83,42),
                   "HeV":(38,167,255),"Fuel":(80,83,239)}

        for tid, v, (x1,y1,x2,y2) in active:
            color = CONFIG["class_colors"].get(v["type"],(180,180,180))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            pts = v["positions"][-15:]
            for i in range(1,len(pts)):
                a = i/len(pts)
                cv2.line(frame,pts[i-1],pts[i],tuple(int(c*a) for c in color),1)
            label = f"#{tid} {v['type']} [{v['engine']}]"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(frame,label,(x1+2,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(20,20,20),1)
            cv2.circle(frame,(x2-8,y1+8),6,dot_bgr.get(v["engine"],(180,180,180)),-1)

        # HUD
        remain = max(0, CONFIG["window_min"]*60-(time.time()-self.window_start_t))
        hud = [f"Frame:{self.frame_count}",
               f"Active:{len(active)}",
               f"Counted:{self.total_counted}",
               f"Window:{int(remain//60):02d}:{int(remain%60):02d} left"]
        cv2.rectangle(frame,(0,0),(200,20+len(hud)*20),(15,15,15),-1)
        for i,line in enumerate(hud):
            col = (0,255,120) if "Counted" in line else (200,200,200)
            cv2.putText(frame,line,(8,18+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

        # Engine sidebar
        y=18
        cv2.rectangle(frame,(w-155,0),(w,20+len(self.window_stats)*18),(15,15,15),-1)
        cv2.putText(frame,"Engine mix:",(w-150,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)
        for eng,count in sorted(self.window_stats.items()):
            y+=18
            cv2.circle(frame,(w-145,y-4),5,dot_bgr.get(eng,(180,180,180)),-1)
            cv2.putText(frame,f"{eng}:{count}",(w-136,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
        return frame

    def run_video(self, source, output=None, show=True):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            print(f"[Error] Cannot open: {source}"); return
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip  = max(1, int(fps/CONFIG["fps_target"]))
        print(f"[Video] {W}x{H}@{fps:.0f}fps — processing every {skip} frame(s)\n")
        writer = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,(W,H)) if output else None
        fc = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            fc += 1
            if fc % skip != 0: continue
            ts = fc / fps
            annotated, _ = self.process_frame(frame, ts)
            if writer: writer.write(annotated)
            if show:
                cv2.imshow("Highway Pollution System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
        self.estimator.close_window()
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("\n[Done]", self.estimator.get_summary())

    def run_demo(self):
        print("[Demo] Simulating highway traffic...\n")
        vehicles = [
            ("sedan","blue","BEV"),("suv","standard","Fuel"),
            ("bus","blue","BEV"),("truck","standard","Fuel"),
            ("sedan","standard","HeV"),("suv","standard","Fuel"),
            ("pickup","standard","Fuel"),("sedan","green","FCEV"),
            ("van","standard","Fuel"),("bus","standard","Fuel"),
        ] * 3
        for i, (vtype, pcolor, expected) in enumerate(vehicles):
            eng = self.engine_id.identify(pcolor, vehicle_type=vtype)
            self.estimator.add_vehicle(vtype, eng.engine_type)
            self.total_counted += 1
            print(f"  [{i+1:2d}] {vtype:8s} plate={pcolor:10s} "
                  f"→ {eng.engine_type:5s} CO2={eng.co2_g_per_km:>5.0f}g/km")
        report = self.estimator.close_window()
        report.print_report()
        print("\n[Summary]", self.estimator.get_summary())
        return report


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def start_dashboard(port: int = 8000):
    """Start the FastAPI dashboard in a background thread."""
    def run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(2)
    print(f"[Dashboard] Running at http://localhost:{port}")
    print(f"[Dashboard] Open your browser at: http://127.0.0.1:{port}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Highway Pollution Prediction System")
    parser.add_argument("--source",    default=None,
        help="Video: file path | camera index (0) | rtsp://...")
    parser.add_argument("--weights",   default="yolov8n.pt",
        help="YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--output",    default=None,
        help="Save annotated video to this path")
    parser.add_argument("--demo",      action="store_true",
        help="Run demo mode (no video needed)")
    parser.add_argument("--dashboard", action="store_true",
        help="Launch dashboard only (no video)")
    parser.add_argument("--no-show",   action="store_true",
        help="Don't show video window")
    parser.add_argument("--port",      type=int, default=8000,
        help="Dashboard port (default: 8000)")
    args = parser.parse_args()

    CONFIG["weights"] = args.weights

    # Always start dashboard in background
    start_dashboard(args.port)

    if args.dashboard:
        print("[Mode] Dashboard only — press Ctrl+C to stop")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Stopped]")
        return

    # Build pipeline
    pipeline = HighwayPollutionPipeline()

    if args.demo or args.source is None:
        print("[Mode] Demo simulation\n")
        pipeline.run_demo()
        print("\n[Dashboard still running] Open http://127.0.0.1:8000")
        print("Press Ctrl+C to stop...")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Stopped]")
    else:
        print(f"[Mode] Video: {args.source}\n")
        pipeline.run_video(
            source = args.source,
            output = args.output,
            show   = not args.no_show,
        )


if __name__ == "__main__":
    main()
