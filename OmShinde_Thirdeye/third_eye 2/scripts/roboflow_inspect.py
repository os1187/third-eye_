#!/usr/bin/env python3
"""
Roboflow Account Inspector

Lists your Roboflow workspaces, projects, and versions via the public API.

Usage:
  export ROBOFLOW_API_KEY=...  # or pass --api-key
  python scripts/roboflow_inspect.py                 # list workspaces
  python scripts/roboflow_inspect.py --workspace WS  # list projects in workspace
  python scripts/roboflow_inspect.py --workspace WS --project PROJ  # list versions

Notes:
  - Recommended export format for YOLO here is "yolov8" (Ultralytics)
  - Workspace and project slugs are visible in Roboflow URLs:
      https://app.roboflow.com/<workspace>/<project>/...
"""

import os
import sys
import argparse
import json
import urllib.parse
import http.client


API_BASE = "api.roboflow.com"


def _encode_path(path: str) -> str:
    # Encode each path segment to safely handle spaces and special chars
    parts = [urllib.parse.quote(p, safe="") for p in path.split("/") if p]
    return "/".join(parts)


def api_get(path: str, api_key: str):
    conn = http.client.HTTPSConnection(API_BASE)
    encoded = _encode_path(path)
    url = f"/{encoded}?api_key={urllib.parse.quote(api_key)}" if encoded else f"/?api_key={urllib.parse.quote(api_key)}"
    conn.request("GET", url)
    resp = conn.getresponse()
    data = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status} {resp.reason}: {data.decode('utf-8', 'ignore')}")
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data.decode("utf-8", "ignore")}


def list_workspaces(api_key: str):
    # Prefer explicit endpoint per API hint
    data = api_get("workspaces", api_key)
    workspaces = data.get("workspaces", []) if isinstance(data, dict) else data
    print("Workspaces:")
    for ws in workspaces:
        # ws contains {id, name, slug}
        print(f"- name={ws.get('name')} slug={ws.get('slug')} id={ws.get('id')}")


def list_projects(workspace: str, api_key: str):
    data = api_get(f"workspaces/{workspace}", api_key)
    projects = data.get("projects", []) if isinstance(data, dict) else []
    print(f"Projects in workspace '{workspace}':")
    for p in projects:
        # p contains {id, name, slug}
        print(f"- name={p.get('name')} slug={p.get('slug')} id={p.get('id')}")


def list_versions(workspace: str, project: str, api_key: str):
    data = api_get(f"workspaces/{workspace}/projects/{project}", api_key)
    versions = data.get("versions") or []
    print(f"Versions for project '{workspace}/{project}':")
    for v in versions:
        # v contains {id, version, created, images, annotations, splits}
        print(
            f"- v{v.get('version')} id={v.get('id')} images={v.get('images')} annotations={v.get('annotations')}"
        )
    print("\nRecommended export format for YOLO: 'yolov8'\nOther common formats: 'yolov5', 'coco', 'voc', 'tfrecord' (conversion needed)")


def main():
    parser = argparse.ArgumentParser(description="Inspect Roboflow account resources")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--workspace", help="Workspace slug")
    parser.add_argument("--project", help="Project slug")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Provide --api-key or set ROBOFLOW_API_KEY env var.")
        sys.exit(1)

    try:
        if not args.workspace:
            list_workspaces(args.api_key)
        elif not args.project:
            list_projects(args.workspace, args.api_key)
        else:
            list_versions(args.workspace, args.project, args.api_key)
    except Exception as e:
        print(f"Failed to query Roboflow API: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()