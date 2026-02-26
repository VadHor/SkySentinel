#!/usr/bin/env python3
"""
SkySentinel — main.py (v2 — Tri-Filtre)
=========================================
Surveillance du ciel genevois en temps réel avec triple corrélation :

  Check 1 : Radar (OpenSky Network — ADS-B)
  Check 2 : Satellite (Skyfield — TLE propagation)
  Check 3 : Vitesse angulaire (classification du mouvement)

Architecture
------------
  ┌──────────────┐                    ┌─────────────────────────────────┐
  │ VideoThread  │     asyncio.Queue  │     TriFilterCorrelator         │
  │  (OpenCV)    │ ─────────────────▸ │                                 │
  │  MOG2+Mask   │                    │  ┌─────────┐  ┌─────────────┐  │
  └──────────────┘                    │  │ Radar   │  │ Satellite   │  │
                                      │  │ OpenSky │  │ Skyfield    │  │
                                      │  └────┬────┘  └──────┬──────┘  │
                                      │       └───┬──────────┘         │
                                      │           ▼                    │
                                      │    Speed Classifier            │
                                      │    (vitesse angulaire)         │
                                      │           │                    │
                                      │           ▼                    │
                                      │  ┌────────────────┐           │
                                      │  │ Verdict final  │           │
                                      │  │ AIRCRAFT │ SAT │           │
                                      │  │ UNKNOWN  │ LOG │           │
                                      │  └────────────────┘           │
                                      └─────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
from dotenv import load_dotenv

from src.vision_engine import FrameAnalysis, VisionConfig, VisionEngine
from src.space_handler import SpaceConfig, SpaceHandler, SpaceCheckResult, SatelliteMatch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("skysentinel.main")


@dataclass(frozen=True)
class AppConfig:
    """Configuration applicative chargée depuis .env."""

    # Flux vidéo HLS
    hls_url: str = os.getenv(
        "HLS_URL",
        "https://webcam.switch.ch/geneva/cam1/playlist.m3u8",
    )

    # OpenSky bounding box autour de Genève
    opensky_url: str = os.getenv(
        "OPENSKY_URL", "https://opensky-network.org/api/states/all"
    )
    bbox_lat_min: float = float(os.getenv("BBOX_LAT_MIN", "46.12"))
    bbox_lat_max: float = float(os.getenv("BBOX_LAT_MAX", "46.35"))
    bbox_lon_min: float = float(os.getenv("BBOX_LON_MIN", "5.90"))
    bbox_lon_max: float = float(os.getenv("BBOX_LON_MAX", "6.30"))

    # OpenSky credentials (optionnel)
    opensky_user: Optional[str] = os.getenv("OPENSKY_USER")
    opensky_pass: Optional[str] = os.getenv("OPENSKY_PASS")

    # Analyse vidéo
    frame_skip: int = int(os.getenv("FRAME_SKIP", "5"))
    radar_timeout_sec: float = float(os.getenv("RADAR_TIMEOUT", "10"))
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "32"))

    # Captures
    alert_capture_dir: str = os.getenv("ALERT_CAPTURE_DIR", "captures")

    # Observateur Genève (pour SpaceHandler)
    observer_lat: float = float(os.getenv("OBSERVER_LAT", "46.20"))
    observer_lon: float = float(os.getenv("OBSERVER_LON", "6.14"))

    # Conversion px → degrés (approximation, à calibrer par webcam)
    cam_fov_h_deg: float = float(os.getenv("CAM_FOV_H_DEG", "90"))
    cam_fov_v_deg: float = float(os.getenv("CAM_FOV_V_DEG", "60"))


CFG = AppConfig()


# ---------------------------------------------------------------------------
# Verdict du Tri-Filtre
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    AIRCRAFT = "AIRCRAFT"
    SATELLITE = "SATELLITE"
    UNKNOWN_HIGH = "UNKNOWN_HIGH"   # Alerte haute priorité
    UNKNOWN_LOW = "UNKNOWN_LOW"     # Pas identifié mais vitesse cohérente


@dataclass
class TriFilterResult:
    """Résultat complet de la corrélation tri-filtre."""
    verdict: Verdict
    analysis: FrameAnalysis
    # Check 1 — Radar
    radar_flights: list[dict]
    radar_matched: bool
    # Check 2 — Satellite
    space_result: Optional[SpaceCheckResult]
    satellite_matched: Optional[SatelliteMatch]
    # Check 3 — Vitesse
    speed_classification: str  # "aircraft", "satellite", "unknown"
    angular_speed_est_deg_s: float


# ---------------------------------------------------------------------------
# Thread vidéo
# ---------------------------------------------------------------------------
class VideoThread(threading.Thread):
    """Thread dédié à la capture HLS + analyse VisionEngine."""

    def __init__(
        self,
        detection_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="VideoThread", daemon=True)
        self.queue = detection_queue
        self.loop = loop
        self.stop_event = stop_event

        vision_cfg = VisionConfig(alert_capture_dir=CFG.alert_capture_dir)
        self.engine = VisionEngine(vision_cfg)
        self._frame_count = 0
        self._detection_count = 0

    def run(self) -> None:
        logger.info("▶ VideoThread démarré — flux : %s", CFG.hls_url)
        cap = cv2.VideoCapture(CFG.hls_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("❌ Impossible d'ouvrir le flux HLS : %s", CFG.hls_url)
            return

        try:
            self._capture_loop(cap)
        except Exception:
            logger.exception("💥 Erreur fatale dans VideoThread")
        finally:
            cap.release()
            logger.info(
                "⏹ VideoThread arrêté — %d frames / %d détections",
                self._frame_count, self._detection_count,
            )

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        raw_count = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Frame perdue — reconnexion…")
                time.sleep(1)
                cap.open(CFG.hls_url, cv2.CAP_FFMPEG)
                continue

            raw_count += 1
            if raw_count % CFG.frame_skip != 0:
                continue

            self._frame_count += 1
            analysis = self.engine.analyze_frame(frame, keep_frames=True)

            if analysis.movement_detected:
                self._detection_count += 1
                asyncio.run_coroutine_threadsafe(
                    self._enqueue(analysis), self.loop
                )

    async def _enqueue(self, analysis: FrameAnalysis) -> None:
        try:
            self.queue.put_nowait(analysis)
        except asyncio.QueueFull:
            logger.warning("⚠ Queue pleine — détection ignorée")


# ---------------------------------------------------------------------------
# Corrélateur Tri-Filtre (async)
# ---------------------------------------------------------------------------
class TriFilterCorrelator:
    """
    Orchestre les 3 checks en parallèle (asyncio.gather) pour chaque
    détection, puis rend un verdict final.
    """

    def __init__(
        self,
        detection_queue: asyncio.Queue,
        engine: VisionEngine,
        space_handler: SpaceHandler,
    ) -> None:
        self.queue = detection_queue
        self.engine = engine
        self.space = space_handler
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            "total": 0,
            "aircraft": 0,
            "satellite": 0,
            "unknown_high": 0,
            "unknown_low": 0,
            "errors": 0,
        }

    async def start(self, stop_event: threading.Event) -> None:
        """Boucle principale du corrélateur."""
        logger.info("▶ TriFilterCorrelator démarré")

        auth = None
        if CFG.opensky_user and CFG.opensky_pass:
            auth = aiohttp.BasicAuth(CFG.opensky_user, CFG.opensky_pass)
            logger.info("🔑 Authentification OpenSky activée")

        async with aiohttp.ClientSession(auth=auth) as session:
            self._session = session

            # Charger les TLE au démarrage
            await self.space.ensure_tle_loaded(session)

            while not stop_event.is_set():
                try:
                    analysis = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Rafraîchir les TLE périodiquement
                    if self.space.tle_age_hours > 6:
                        await self.space.ensure_tle_loaded(session)
                    continue

                await self._run_tri_filter(analysis)

        logger.info("⏹ TriFilterCorrelator arrêté — stats=%s",
                     json.dumps(self._stats, indent=2))

    async def _run_tri_filter(self, analysis: FrameAnalysis) -> None:
        """Exécute les 3 checks en parallèle et rend un verdict."""
        self._stats["total"] += 1

        try:
            # ===== CHECK 1 + CHECK 2 en parallèle =====
            radar_task = asyncio.create_task(self._check_radar())
            space_task = asyncio.create_task(
                self.space.check_satellites(analysis.timestamp)
            )

            flights, space_result = await asyncio.gather(
                radar_task, space_task, return_exceptions=True
            )

            # Gérer les exceptions individuelles
            if isinstance(flights, Exception):
                logger.error("Check Radar échoué : %s", flights)
                flights = []
            if isinstance(space_result, Exception):
                logger.error("Check Satellite échoué : %s", space_result)
                space_result = None

            # ===== CHECK 3 — Vitesse angulaire =====
            angular_speed = self._estimate_angular_speed(analysis)
            speed_class = self.space.classify_angular_speed(angular_speed)

            # Corrélation spatiale satellite ↔ détection visuelle
            sat_match = None
            if isinstance(space_result, SpaceCheckResult) and space_result.satellites_in_fov:
                det_az, det_el = self._pixel_to_azel(
                    analysis.centroid_x_pct, analysis.centroid_y_pct
                )
                sat_match = self.space.correlate_with_detection(
                    det_az, det_el, space_result
                )

            # ===== VERDICT FINAL =====
            verdict = self._determine_verdict(
                radar_matched=bool(flights) if isinstance(flights, list) else False,
                satellite_matched=sat_match,
                speed_class=speed_class,
            )

            result = TriFilterResult(
                verdict=verdict,
                analysis=analysis,
                radar_flights=flights if isinstance(flights, list) else [],
                radar_matched=bool(flights) if isinstance(flights, list) else False,
                space_result=space_result if isinstance(space_result, SpaceCheckResult) else None,
                satellite_matched=sat_match,
                speed_classification=speed_class,
                angular_speed_est_deg_s=angular_speed,
            )

            self._handle_verdict(result)

        except Exception:
            self._stats["errors"] += 1
            logger.exception("Erreur dans le tri-filtre")

    def _determine_verdict(
        self,
        radar_matched: bool,
        satellite_matched: Optional[SatelliteMatch],
        speed_class: str,
    ) -> Verdict:
        """
        Arbre de décision du tri-filtre :

        1. Radar match          → AIRCRAFT
        2. Satellite match      → SATELLITE
        3. Speed = "aircraft"   → UNKNOWN_LOW  (drone? petit avion non-ADS-B?)
        4. Speed = "satellite"  → UNKNOWN_HIGH (objet orbital non catalogué!)
        5. Speed = "unknown"    → UNKNOWN_HIGH
        """
        if radar_matched:
            return Verdict.AIRCRAFT

        if satellite_matched is not None:
            return Verdict.SATELLITE

        if speed_class == "aircraft":
            return Verdict.UNKNOWN_LOW

        # satellite-speed sans TLE match, ou vitesse inclassifiable
        return Verdict.UNKNOWN_HIGH

    def _handle_verdict(self, result: TriFilterResult) -> None:
        """Log et actions selon le verdict."""
        v = result.verdict
        key = v.value.lower()
        self._stats[key] = self._stats.get(key, 0) + 1

        if v == Verdict.AIRCRAFT:
            callsigns = [f.get("callsign", "?").strip() for f in result.radar_flights]
            logger.info(
                "✈  AIRCRAFT — %d vol(s) : %s | ω=%.3f°/s",
                len(result.radar_flights),
                ", ".join(callsigns[:5]),
                result.angular_speed_est_deg_s,
            )

        elif v == Verdict.SATELLITE:
            sat = result.satellite_matched
            logger.info(
                "🛰  SATELLITE — %s (NORAD #%d) | Az=%.1f° El=%.1f° | "
                "dist=%.0fkm | ω=%.3f°/s",
                sat.name, sat.catalog_number,
                sat.azimuth_deg, sat.elevation_deg,
                sat.distance_km, sat.angular_speed_deg_per_sec,
            )

        elif v == Verdict.UNKNOWN_HIGH:
            logger.warning(
                "🚨 ALERTE HAUTE — Objet NON IDENTIFIÉ | speed_class=%s | "
                "ω=%.3f°/s | radar=❌ satellite=❌",
                result.speed_classification,
                result.angular_speed_est_deg_s,
            )
            if result.analysis.frame_raw is not None:
                self.engine.save_alert_capture(
                    result.analysis.frame_raw, label="UNKNOWN_HIGH"
                )

        elif v == Verdict.UNKNOWN_LOW:
            logger.info(
                "❓ UNKNOWN_LOW — Vitesse avion sans OpenSky | ω=%.3f°/s "
                "(drone? ULM? avion privé?)",
                result.angular_speed_est_deg_s,
            )

        # Toujours logger dans le fichier
        self._log_event(result)

    def _estimate_angular_speed(self, analysis: FrameAnalysis) -> float:
        """Convertit la vitesse px/s en °/s via le FOV caméra."""
        if analysis.motion_speed_px_per_sec <= 0:
            return 0.0
        # Résolution estimée — TODO: récupérer du flux réel
        assumed_width_px = 1920.0
        deg_per_px = CFG.cam_fov_h_deg / assumed_width_px
        return analysis.motion_speed_px_per_sec * deg_per_px

    def _pixel_to_azel(self, x_pct: float, y_pct: float) -> tuple[float, float]:
        """Convertit une position pixel (%) en azimut/élévation."""
        sc = self.space.cfg
        az = sc.cam_az_center_deg + (x_pct - 0.5) * sc.cam_az_fov_deg
        az = az % 360
        el = sc.cam_el_max_deg - y_pct * (sc.cam_el_max_deg - sc.cam_el_min_deg)
        return az, el

    # -----------------------------------------------------------------------
    # Radar check
    # -----------------------------------------------------------------------
    async def _check_radar(self) -> list[dict]:
        """Interroge l'API OpenSky avec la bounding box Genève."""
        params = {
            "lamin": CFG.bbox_lat_min,
            "lamax": CFG.bbox_lat_max,
            "lomin": CFG.bbox_lon_min,
            "lomax": CFG.bbox_lon_max,
        }
        assert self._session is not None
        try:
            async with self._session.get(
                CFG.opensky_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=CFG.radar_timeout_sec),
            ) as resp:
                if resp.status != 200:
                    logger.warning("OpenSky HTTP %d", resp.status)
                    return []
                data = await resp.json()
                states = data.get("states") or []
                keys = [
                    "icao24", "callsign", "origin_country", "time_position",
                    "last_contact", "longitude", "latitude", "baro_altitude",
                    "on_ground", "velocity", "true_track", "vertical_rate",
                    "sensors", "geo_altitude", "squawk", "spi",
                    "position_source", "category",
                ]
                return [
                    dict(zip(keys, s)) for s in states
                    if not dict(zip(keys, s)).get("on_ground", True)
                ]
        except asyncio.TimeoutError:
            logger.warning("OpenSky timeout (%ss)", CFG.radar_timeout_sec)
            return []

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    @staticmethod
    def _log_event(result: TriFilterResult) -> None:
        """Persiste chaque événement dans events_log.jsonl."""
        log_path = Path("events_log.jsonl")
        record = {
            "timestamp": result.analysis.timestamp,
            "logged_at": time.time(),
            "verdict": result.verdict.value,
            "speed_class": result.speed_classification,
            "angular_speed_deg_s": round(result.angular_speed_est_deg_s, 4),
            "centroid_pct": [
                round(result.analysis.centroid_x_pct, 3),
                round(result.analysis.centroid_y_pct, 3),
            ],
            "contours": result.analysis.contours_count,
            "radar": {
                "matched": result.radar_matched,
                "count": len(result.radar_flights),
                "flights": [
                    {
                        "icao24": f.get("icao24"),
                        "callsign": (f.get("callsign") or "").strip(),
                        "altitude_m": f.get("geo_altitude"),
                        "velocity_ms": f.get("velocity"),
                    }
                    for f in result.radar_flights[:10]
                ],
            },
            "satellite": None,
        }
        if result.satellite_matched:
            sat = result.satellite_matched
            record["satellite"] = {
                "name": sat.name,
                "norad": sat.catalog_number,
                "az_deg": round(sat.azimuth_deg, 2),
                "el_deg": round(sat.elevation_deg, 2),
                "dist_km": round(sat.distance_km, 1),
                "angular_speed": round(sat.angular_speed_deg_per_sec, 4),
                "illuminated": sat.is_illuminated,
            }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------
async def main() -> None:
    """Point d'entrée — orchestre VideoThread + TriFilterCorrelator."""
    logger.info("=" * 65)
    logger.info("  🛡  SkySentinel v2 — Tri-Filtre (Radar + Satellite + Speed)")
    logger.info("=" * 65)
    logger.info("  HLS         : %s", CFG.hls_url)
    logger.info("  BBox Radar  : (%.2f, %.2f) → (%.2f, %.2f)",
                CFG.bbox_lat_min, CFG.bbox_lon_min,
                CFG.bbox_lat_max, CFG.bbox_lon_max)
    logger.info("  Observateur : %.4f°N %.4f°E (Genève)",
                CFG.observer_lat, CFG.observer_lon)
    logger.info("  FrameSkip   : %d", CFG.frame_skip)
    logger.info("  CAM FOV     : %.0f° × %.0f°", CFG.cam_fov_h_deg, CFG.cam_fov_v_deg)
    logger.info("=" * 65)

    stop_event = threading.Event()
    detection_queue: asyncio.Queue[FrameAnalysis] = asyncio.Queue(
        maxsize=CFG.max_queue_size
    )
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _shutdown(stop_event))

    # SpaceHandler
    space_cfg = SpaceConfig(
        observer_lat=CFG.observer_lat,
        observer_lon=CFG.observer_lon,
    )
    space_handler = SpaceHandler(space_cfg)

    # VideoThread
    video_thread = VideoThread(detection_queue, loop, stop_event)
    video_thread.start()

    # TriFilterCorrelator
    correlator = TriFilterCorrelator(
        detection_queue, video_thread.engine, space_handler
    )
    await correlator.start(stop_event)

    video_thread.join(timeout=5)
    logger.info("🏁 SkySentinel v2 arrêté proprement.")


def _shutdown(stop_event: threading.Event) -> None:
    logger.info("⏳ Signal d'arrêt reçu — extinction en cours…")
    stop_event.set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interruption clavier — bye !")
        sys.exit(0)
