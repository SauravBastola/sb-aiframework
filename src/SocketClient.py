import socket
import os
import json

UNIXSOCKET_PATH = "/tmp/ai-framework.sock"


class SocketClient:
    def __init__(self):
        self.path = UNIXSOCKET_PATH
        self.socket = None
        self.connect()

    def connect(self):
        if self.socket is not None:
            raise RuntimeError("Already connected")

        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(self.path)
        print(f"Connected to {self.path}")

    def req(self, path):
        unix_path = path.replace("/", ":")
        self.send_message(unix_path)
        response = self.receive_message()
        return json.loads(response)

    def send_message(self, message):
        if self.socket is None:
            raise RuntimeError("Not connected")

        self.socket.sendall(message.encode())

    def receive_message(self):
        if self.socket is None:
            raise RuntimeError("Not connected")

        response = self.socket.recv(2048).decode()
        return response

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Connection closed")
