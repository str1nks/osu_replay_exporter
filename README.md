1. Установка (в cmd на Windows):
pip install pyinstaller pygame numpy imageio-ffmpeg

2. Сборка (в папке со скриптом):
pyinstaller --onedir --name OsuReplayExporter --collect-all imageio_ffmpeg --collect-all pygame osu_replay_exporter.py