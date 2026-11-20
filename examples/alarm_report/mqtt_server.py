"""
Local MQTT broker for testing AlarmReport.

  python mqtt_server.py

Then in demo.py:

  AlarmReport(..., mqtt_broker="127.0.0.1", mqtt_port=1883, mqtt_http_base="http://127.0.0.1:9966")
"""
import argparse
import json
import os
import socket
import threading
import urllib.request


def encode_len(n):
    out = bytearray()
    while True:
        b = n % 128
        n //= 128
        if n:
            b |= 0x80
        out.append(b)
        if not n:
            break
    return bytes(out)


def read_len(sock):
    multiplier, value = 1, 0
    while True:
        b = sock.recv(1)
        if not b:
            return None
        value += (b[0] & 127) * multiplier
        if b[0] < 128:
            return value
        multiplier *= 128


def read_packet(sock):
    header = sock.recv(1)
    if not header:
        return None
    remaining = read_len(sock)
    if remaining is None:
        return None
    body = b""
    while len(body) < remaining:
        chunk = sock.recv(remaining - len(body))
        if not chunk:
            return None
        body += chunk
    return header[0], body


def mqtt_str(data, i):
    n = int.from_bytes(data[i:i + 2], "big")
    s = data[i + 2:i + 2 + n].decode("utf-8", errors="replace")
    return s, i + 2 + n


def topic_match(filt, topic):
    fp, tp = filt.split("/"), topic.split("/")
    for i, f in enumerate(fp):
        if f == "#":
            return True
        if i >= len(tp) or (f != "+" and f != tp[i]):
            return False
    return len(fp) == len(tp)


class Broker:
    def __init__(self, on_publish):
        self.on_publish = on_publish
        self.lock = threading.Lock()
        self.subs = []  # (sock, topic_filter)

    def add_sub(self, sock, filt):
        with self.lock:
            self.subs.append((sock, filt))

    def drop(self, sock):
        with self.lock:
            self.subs = [s for s in self.subs if s[0] is not sock]

    def forward(self, topic, payload, qos):
        self.on_publish(topic, payload)
        packet = self._publish_packet(topic, payload, qos)
        with self.lock:
            targets = [s for s in list(self.subs) if topic_match(s[1], topic)]
        for sock, _ in targets:
            try:
                sock.sendall(packet)
            except OSError:
                self.drop(sock)

    @staticmethod
    def _publish_packet(topic, payload, qos):
        topic_b = topic.encode("utf-8")
        body = len(topic_b).to_bytes(2, "big") + topic_b
        if qos:
            body += b"\x00\x01"
        body += payload if isinstance(payload, bytes) else payload.encode("utf-8")
        flags = 0x30 | (qos << 1)
        return bytes([flags]) + encode_len(len(body)) + body


def handle_client(sock, broker):
    try:
        while True:
            pkt = read_packet(sock)
            if pkt is None:
                break
            first, body = pkt
            kind = first >> 4
            if kind == 1:  # CONNECT
                sock.sendall(b"\x20\x02\x00\x00")
            elif kind == 3:  # PUBLISH
                qos = (first >> 1) & 3
                topic, i = mqtt_str(body, 0)
                pid = None
                if qos:
                    pid = body[i:i + 2]
                    i += 2
                broker.forward(topic, body[i:], qos)
                if qos and pid:
                    sock.sendall(b"\x40\x02" + pid)
            elif kind == 8:  # SUBSCRIBE
                pid = body[:2]
                i = 2
                codes = bytearray()
                while i < len(body):
                    topic, i = mqtt_str(body, i)
                    qos = body[i]
                    i += 1
                    broker.add_sub(sock, topic)
                    codes.append(min(qos, 1))
                ack = pid + bytes(codes)
                sock.sendall(b"\x90" + encode_len(len(ack)) + ack)
            elif kind == 12:  # PINGREQ
                sock.sendall(b"\xd0\x00")
            elif kind == 14:  # DISCONNECT
                break
    finally:
        broker.drop(sock)
        sock.close()


def save_url(url, save_dir, http_base):
    if not url:
        return
    if url.startswith("/"):
        url = http_base.rstrip("/") + url
    name = url.rstrip("/").split("/")[-1]
    path = os.path.join(save_dir, name)
    try:
        with urllib.request.urlopen(url, timeout=20) as r, open(path, "wb") as f:
            f.write(r.read())
        print(f"  saved {path} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        print(f"  download failed {url}: {e}")


def main():
    parser = argparse.ArgumentParser(description="MQTT broker for AlarmReport test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--http-base", default="http://127.0.0.1:9966", help="used when image_url/video_url is a relative path")
    parser.add_argument("--save-dir", default="./mqtt_recv")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    def on_publish(topic, payload):
        text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
        print(f"\n[{topic}] {text}")
        try:
            data = json.loads(text)
        except Exception:
            return
        save_url(data.get("image_url"), args.save_dir, args.http_base)
        save_url(data.get("video_url"), args.save_dir, args.http_base)

    broker = Broker(on_publish)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen()
    print(f"MQTT broker {args.host}:{args.port}, save files to {args.save_dir}")
    print('AlarmReport(..., mqtt_broker="127.0.0.1", mqtt_http_base="http://127.0.0.1:9966")')
    while True:
        sock, addr = server.accept()
        print(f"client {addr[0]}:{addr[1]}")
        threading.Thread(target=handle_client, args=(sock, broker), daemon=True).start()


if __name__ == "__main__":
    main()
