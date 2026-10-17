"""Запуск дашборда из сохранённой конфигурации (как в практике по Docker-развёртыванию).

    python lab6_dashboard/run_dashboard.py          # http://localhost:9050
Порт можно поменять переменной окружения PORT.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# explainerdashboard 0.5.8 импортирует pkg_resources, которого нет в setuptools >= 81.
# В requirements зафиксирован setuptools<81; эта заглушка — на случай, если окружение собрано иначе.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types
    from packaging.version import parse as _parse_version
    _pr = types.ModuleType("pkg_resources")
    _pr.parse_version = _parse_version
    sys.modules["pkg_resources"] = _pr

from explainerdashboard import ExplainerDashboard  # noqa: E402

CONFIG = HERE / "artifacts" / "dashboard.yaml"

if __name__ == "__main__":
    if not CONFIG.exists():
        sys.exit("Нет artifacts/dashboard.yaml — сначала выполните: python lab6_dashboard/build_dashboard.py")
    os.chdir(CONFIG.parent)  # explainer.joblib лежит рядом с yaml
    db = ExplainerDashboard.from_config(CONFIG.name)
    port = int(os.environ.get("PORT", 9050))
    print(f"Дашборд: http://localhost:{port}")
    db.run(host="0.0.0.0", port=port, use_waitress=True)
