from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from utils.logger import ensure_dir


def export_run(
    export_dir: str,
    filename_template: str,
    payload: Dict[str, Any],
    prompt_version: str,
    index_version: str,
    safety_ruleset_version: str,
) -> Path:
    ensure_dir(export_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = filename_template.format(timestamp=timestamp)
    path = Path(export_dir) / filename

    ordered_payload = OrderedDict()
    ordered_payload["timestamp"] = timestamp
    ordered_payload["prompt_version"] = prompt_version
    ordered_payload["index_version"] = index_version
    ordered_payload["safety_ruleset_version"] = safety_ruleset_version
    for key in sorted(payload.keys()):
        ordered_payload[key] = payload[key]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(ordered_payload, handle, ensure_ascii=False, indent=2)
    return path

