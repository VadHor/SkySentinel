# 🛰️ SkySentinel : Geneva Aerial Surveillance POC

**SkySentinel** est une preuve de concept (POC) d'intelligence géospatiale conçue pour surveiller le ciel de Genève en temps réel. Le système croise l'analyse de flux vidéo publics (Computer Vision) avec les données radar mondiales (ADS-B) pour identifier les activités aériennes non répertoriées.

L'idée est de créer un "filtre intelligent" pour l'observation du ciel :
1. **Détecter** : Identifier tout mouvement dans la zone aérienne via des webcams HD.
2. **Identifier** : Interroger instantanément les API radar pour voir si un vol commercial est présent.
3. **Alerter** : Si un mouvement est détecté mais qu'aucun avion n'est répertorié, le système capture une preuve visuelle.

* **Langage :** Python 3.9+
* **Vision par ordinateur :** `OpenCV` (Soustraction de fond MOG2, masquage de zone).
* **Data Fusion :** `Requests` pour l'API REST d'OpenSky Network (données ADS-B).
* **Interface :** `Streamlit` (en cours) pour la visualisation en direct.
* **Automatisation :** `Python-dotenv` pour la gestion des configurations et secrets.

Structure du Repository
```text
SkySentinel/
├── src/
│   ├── vision_engine.py  # Analyse du flux vidéo et détection de mouvement
│   ├── radar_handler.py  # Interface avec l'API OpenSky Network
│   └── alerts.py         # Système de notification (Telegram/Discord)
├── data/                 # Dossier de stockage des captures d'anomalies
├── .env                  # Variables de configuration (URL flux, API keys)
├── requirements.txt      # Dépendances Python
└── main.py               # Point d'entrée du programme
