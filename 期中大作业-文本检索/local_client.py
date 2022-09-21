import socket
import json

client = socket.socket()
client.connect(("127.0.0.1", 9001))
client.send()
rec = client.recv(1024)
print(rec)