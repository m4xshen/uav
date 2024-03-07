#
# Tello Python3 Control Demo
#
# http://www.ryzerobotics.com/
#
# 1/1/2018

import threading
import socket
import sys
import time


host = ""
port = 9000
locaddr = (host, port)


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(locaddr)

# please fill UAV IP address
tello_address1 = ("192.168.137.109", 8889)
tello_address2 = ("192.168.137.211", 8889)

message1 = [
    "command",
    "battery?",
    "takeoff",

    "forward 150",
    "up 50",
    "forward 120",
    "stop",

    "land",
]

message2 = [
    "command",
    "battery?",
    "takeoff",

    "up 130",
    "forward 110",
    "down 130",
    "forward 170",

    "land",
]
delay = [3, 5, 7, 5, 5, 5, 5, 5]


def recv():
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            print("{} : {}".format(server, data.decode(encoding="utf-8")))
        except Exception:
            print("\nExit . . .\n")
            break


# recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()


for i in range(0, len(message1)):
    msg1 = message1[i]
    msg2 = message2[i]
    sock.sendto(msg1.encode("utf-8"), tello_address1)
    sock.sendto(msg2.encode("utf-8"), tello_address2)
    time.sleep(delay[i])
