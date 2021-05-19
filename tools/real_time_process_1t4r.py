import threading as th
import numpy as np
import socket
import DSP_2t4r


class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length, data_address, buff_size):
        """
        :param name: str
                        Object name

        :param bin_data: queue object
                        A queue used to store adc data from udp stream

        :param data_frame_length: int
                        Length of a single frame

        :param data_address: (str, int)
                        Address for binding udp stream, str for host IP address, int for host data port

        :param buff_size: int
                        Socket buffer size
        """
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.buff_size = buff_size

    def run(self):
        # convert bytes to data type int16
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        # array for putting raw data
        np_data = []
        # count frame
        count_frame = 0
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.bind(self.data_address)
        print("Create socket successfully")
        print("Now start data streaming")
        # main loop
        while True:
            data, addr = data_socket.recvfrom(self.buff_size)
            data = data[10:]
            np_data.extend(np.frombuffer(data, dtype=dt))
            # while np_data length exceeds frame length, do following
            if len(np_data) >= self.frame_length:
                count_frame += 1
                # print("Frame No.", count_frame)
                # put one frame data into bin data array
                self.bin_data.put(np_data[0:self.frame_length])
                # remove one frame length data from array
                np_data = np_data[self.frame_length:]


class DataProcessor(th.Thread):
    def __init__(self, name, config, bin_queue, rdi_queue, rai_queue, file_name):
        """
        :param name: str
                        Object name

        :param config: sequence of ints
                        Radar config in the order
                        [0]: samples number
                        [1]: chirps number
                        [3]: transmit antenna number
                        [4]: receive antenna number

        :param bin_queue: queue object
                        A queue for access data received by UdpListener

        :param rdi_queue: queue object
                        A queue for store RDI

        :param rai_queue: queue object
                        A queue for store RDI

        """
        th.Thread.__init__(self, name=name)
        self.adc_sample = config[0]
        self.chirp_num = config[1]
        self.tx_num = config[2]
        self.rx_num = config[3]
        self.bin_queue = bin_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.filename = file_name
        self.weight_matrix = np.zeros([181, 4], dtype=complex)
        self.out_matrix = np.zeros([8192, 181], dtype=complex)
        Fc = 77e9
        count = 0
        lambda_start = 3e8 / Fc
        for theta in range(-90, 91):
            d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
            beamforming_factor = np.array([0, d, 2 * d, 3 * d]) / (3e8 / Fc)
            self.weight_matrix[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor)
            count += 1

    def run(self):
        frame_count = 0
        while True:
            data = self.bin_queue.get()
            data = np.reshape(data, [-1, 4])
            data = data[:, 0:2:] + 1j * data[:, 2::]
            data = np.reshape(data, [self.chirp_num * self.tx_num, -1, self.adc_sample])
            data = data.transpose([0, 2, 1])
            # print(np.shape(data))
            # data = data[0: 64: 2, :, :]
            # ch3_data = data[1: 64: 2, :, :]
            # data = np.concatenate([ch1_data, ch3_data], axis=2)
            frame_count += 1
            rdi_raw, rdi = DSP_2t4r.Range_Doppler(data, mode=2, padding_size=[128, 64])
            # rai = DSP_2t4r.Range_Angle(data, mode=1, padding_size=[128, 64, 32])
            # ===============beamforming===============
            # print(np.shape(rdi_raw))
            rdi_raw = rdi_raw.reshape([-1, 4])
            # print(np.shape(rdi_raw))
            for i in range(8192):
                self.out_matrix[i, :] = np.matmul(self.weight_matrix, rdi_raw[i, :])
            rai = self.out_matrix.reshape([128, 64, -1])

            # rai = np.flip(np.abs(rai), axis=1)
            # rai = np.flip(rai, axis=1)
            rai = np.abs(rai)
            # ===============beamforming===============
            # np.save('../data/Real_time_data/' + self.filename + 'Frame_' + str(frame_count), data)

            self.rdi_queue.put(rdi)
            self.rai_queue.put(rai)
