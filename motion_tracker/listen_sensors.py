import json
import socket

MAGNETOMETER_FILE = "res/magnetometer/magnetometer.csv"
GYROSCOPE_FILE = "res/gyroscope/gyroscope.csv"
ACCELEROMETER_FILE = "res/accelerometer/accelerometer.csv"

af = mf = gf = None


def write_to_file(file, reading):
    data = "{0},{1},{2}\n".format(reading['x'], reading['y'], reading['z'])
    file.write(data)
    pass


def connect_to_sensor(ip, port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind((ip, port))
    soc.listen(1)
    print("waiting for connection...")
    conn, addr = soc.accept()
    conn.settimeout(1)
    print('connection established, reading data')

    mf = open(MAGNETOMETER_FILE, "w+")
    af = open(ACCELEROMETER_FILE, "w+")
    gf = open(GYROSCOPE_FILE, "w+")
    af.write('x,y,z\n')
    mf.write('x,y,z\n')
    gf.write('x,y,z\n')

    rec_count = 0
    while True:
        try:
            data = conn.recv(1024)
            data = data.decode('utf-8')
            sensor_data = json.loads(data)
            write_to_file(mf, sensor_data['magnetometerReading'])
            write_to_file(af, sensor_data['accelerometerReading'])
            write_to_file(gf, sensor_data['gyroscopeReading'])
            print('reading received : ', rec_count)
            rec_count += 1
        except Exception as e:
            print(e)
            print('received : ', rec_count)
            conn.close()
            close_files()
            break


def close_files():
    global af, mf, gf
    af.close()
    mf.close()
    gf.close()


# def write(data):
#     sensor = data['sensorType']
#     x = str(data['x'])
#     y = str(data['y'])
#     z = str(data['z'])
#
#     file = af
#     file.write(x, y, z)


connect_to_sensor('192.168.1.67', 2536)
