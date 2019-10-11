import numpy as np
import socket
import pandas as pd

#ip, port setting
IP = "127.0.0.1"
PORT = 6092
#socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((IP,PORT))

#data decoding
#data save CSV file
while True:
    data, addr = s.recvfrom(65535)
    data_heading = data[0]
    data_lanemarker = data[4]
    data_Left = round((data[8] * (1 / 100)) - 1, 2)
    data_Right = round((data[40] * (1 / 100)) - 1, 2)
    data_Bottom = round((data[72] * (1 / 100)) - 1, 2)
    data_Top = round((data[104] * (1 / 100)) - 1, 2)
    data_yawlate = (data[136] * (1 / 100)) - 5
    data_latralangle = data[138]
    data_latralSpeed = data[137] ** 2
    data_reallatralAccelation = round(data_latralSpeed / data_latralangle, 3)

    if data_lanemarker == 1:
        data_lanemarker = 1
    else:
        data_lanemarker = 0
    frame = [data_heading, data_Left, data_Right, data_Bottom, data_Top, data_yawlate, data_reallatralAccelation,  data_lanemarker]

    df = pd.DataFrame(frame)
    df = np.transpose(df)
    df.to_csv("6_rawdata.csv", mode='a', index=False, header=False)





