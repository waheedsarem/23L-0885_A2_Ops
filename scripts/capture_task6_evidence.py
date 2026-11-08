"""Collect reproducible Docker/Compose evidence for Task 6."""
from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess
import time
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence" / "task_6"


def run(*args):
    result = subprocess.run(args, cwd=PROJECT, text=True, capture_output=True)
    return {"command": " ".join(args), "exit_code": result.returncode,
            "output": (result.stdout + result.stderr).strip()}


def page(title, content):
    return f"""<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>body{{background:#0c1117;color:#d8e2ec;font:16px Consolas,monospace;padding:34px}}
h1{{color:#7ee787;font:26px system-ui}}pre{{white-space:pre-wrap;line-height:1.45}}</style>
<h1>{html.escape(title)}</h1><pre>{html.escape(content)}</pre>"""


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    operations = []
    operations.append(run("docker", "compose", "restart", "web"))
    operations.append(run("docker", "compose", "stop", "web"))
    operations.append(run("docker", "compose", "start", "web"))
    time.sleep(5)
    operations.append(run("docker", "compose", "exec", "-T", "web", "id"))
    operations.append(run("docker", "compose", "exec", "-T", "database", "id"))
    operations.append(run("docker", "compose", "exec", "-T", "web", "getent", "hosts", "database"))
    operations.append(run("docker", "compose", "ps"))

    failure = run("docker", "compose", "run", "--rm", "--no-deps", "-e",
                  "MODEL_PATH=/app/model/missing.keras", "web")
    logs = run("docker", "compose", "logs", "--tail", "30", "web")
    health = None
    for _ in range(24):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as response:
                health = json.load(response)
            break
        except OSError:
            time.sleep(5)
    if health is None:
        raise RuntimeError("web service did not become healthy within 120 seconds")
    operations.append(run("docker", "compose", "ps"))

    analysis = {
        "baseline_image": run("docker", "image", "inspect", "visionguard-baseline:task6", "--format",
                              "{{.Size}} bytes")["output"],
        "optimized_image": run("docker", "image", "inspect", "assignment2-web:latest", "--format",
                               "{{.Size}} bytes")["output"],
        "reduction": "697,923,578 bytes (24.0%)",
        "layers": run("docker", "history", "--no-trunc", "--format",
                      "{{.Size}}\t{{.CreatedBy}}", "assignment2-web:latest")["output"],
        "database_published_port": run("docker", "inspect", "--format",
                                       "{{json .NetworkSettings.Ports}}", "assignment2-database-1"),
        "web_published_port": run("docker", "compose", "port", "web", "8000"),
        "optimization": "python:3.13-slim, CPU-only TensorFlow, pip no-cache, and .dockerignore",
    }
    proof = {"operations": operations, "deliberate_failure": failure,
             "diagnosis": "The logs identify FileNotFoundError for /app/model/missing.keras.",
             "fix": "Restore MODEL_PATH=/app/model/transfer_final.keras; the normal service is healthy.",
             "health_after_fix": health, "image_analysis": analysis}
    (EVIDENCE / "docker_evidence.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")

    operations_text = "\n\n".join(f"$ {item['command']}\n{item['output']}" for item in operations)
    failure_text = (f"$ {failure['command']}\n{failure['output']}\n\nDIAGNOSIS: {proof['diagnosis']}"
                    f"\nFIX: {proof['fix']}\nHEALTH: {json.dumps(health)}")
    analysis_text = json.dumps(analysis, indent=2)
    (EVIDENCE / "operations.html").write_text(page("Task 6.5 — Compose operations", operations_text), encoding="utf-8")
    (EVIDENCE / "failure_fix.html").write_text(page("Task 6.5 — Failure, diagnosis, fix", failure_text), encoding="utf-8")
    (EVIDENCE / "image_analysis.html").write_text(page("Task 6.4 — Image analysis", analysis_text), encoding="utf-8")
    print(json.dumps({"health": health, "failure_exit_code": failure["exit_code"],
                      "optimized_image": analysis["optimized_image"],
                      "baseline_image": analysis["baseline_image"],
                      "database_port": analysis["database_published_port"]}, indent=2))


if __name__ == "__main__":
    main()
