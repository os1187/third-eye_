import socket
import struct
import numpy as np
import cv2
from typing import Iterator, Tuple, Optional


def _recv_all(conn: socket.socket, n: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def stream_rgbd(host: str = "0.0.0.0", port: int = 6000) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Yield grayscale image, raw depth (meters), and depth visualization from a TCP RGB-D sender.

    Protocol:
      - Header: 1 byte `frame_type` + 4 bytes big-endian payload length
      - frame_type 0: depth float32 HxW (meters)
      - frame_type 1: grayscale uint8 HxW

    Yields:
      - gray_img: np.ndarray (H, W) uint8 grayscale
      - depth_map: np.ndarray (H, W) float32 meters or None
      - depth_vis: np.ndarray (H, W, 3) uint8 BGR visualization or None
    """
    DEPTH_W, DEPTH_H = 256, 192

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    conn, _ = s.accept()
    conn.setblocking(True)

    depth_map = None
    depth_vis = None
    gray_img = None

    try:
        while True:
            header = _recv_all(conn, 5)
            if not header:
                break

            frame_type = header[0]
            payload_len = struct.unpack(">I", header[1:5])[0]
            payload = _recv_all(conn, payload_len)
            if not payload:
                break

            if frame_type == 0:
                # Depth float32 DEPTH_H x DEPTH_W (meters)
                try:
                    arr = np.frombuffer(payload, dtype=np.float32)
                    if arr.size == DEPTH_W * DEPTH_H:
                        depth_map = arr.reshape(DEPTH_H, DEPTH_W)
                    else:
                        # Fallback: infer WxH roughly from 4:3 aspect
                        w = int((arr.size * 4 / 3) ** 0.5)
                        h = max(1, arr.size // max(1, w))
                        depth_map = arr.reshape(h, w)
                    dv = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
                except Exception:
                    depth_map = None
                    depth_vis = None
            elif frame_type == 1:
                # Grayscale uint8 HxW
                try:
                    gray = np.frombuffer(payload, dtype=np.uint8)
                    pixels = gray.size
                    if pixels == 480 * 360:
                        gw, gh = 480, 360
                    elif pixels == 960 * 720:
                        gw, gh = 960, 720
                    elif pixels == 1280 * 720:
                        gw, gh = 1280, 720
                    else:
                        gw = int((pixels * 4 / 3) ** 0.5)
                        gh = max(1, pixels // max(1, gw))
                    gray_img = gray.reshape(gh, gw)
                except Exception:
                    gray_img = None

            if gray_img is not None:
                yield gray_img, depth_map, depth_vis
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass