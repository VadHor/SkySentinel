import aiohttp
import logging
from typing import List, Dict, Any

class RadarHandler:
    def __init__(self):
        self.api_url = "https://opensky-network.org/api/states/all"

    async def check_aircraft_in_bbox(self, lamin: float, lomin: float, lamax: float, lomax: float) -> List[Dict[str, Any]]:
        params = {'lamin': lamin, 'lomin': lomin, 'lamax': lamax, 'lomax': lomax}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params, timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        states = data.get('states', [])
                        if not states: return []
                        
                        return [{'callsign': s[1].strip() if s[1] else "UNKNOWN", 'altitude': s[7]} for s in states]
                    return []
        except Exception as e:
            logging.error(f"Radar Error: {e}")
            return []
