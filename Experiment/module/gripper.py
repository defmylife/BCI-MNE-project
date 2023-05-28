import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *                    # Uses Dynamixel SDK library
import time

class Gripper():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ADDR_MX_TORQUE_ENABLE      = 24              
        self.ADDR_MX_GOAL_POSITION      = 30
        self.ADDR_MX_PRESENT_POSITION   = 36
        self.DXL_ID                      = 1                 
        self.BAUDRATE                    = 57600             
        self.DEVICENAME                  = "COM12"    
        self.PROTOCOL_VERSION            = 1.0
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()

        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_MX_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")

        self.open_pos = 3150
        self.close_pos = 2445

    def close_gripper(self):
        # while 1:
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_MX_GOAL_POSITION, self.close_pos)
            # dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_MX_PRESENT_POSITION)
            # print(dxl_present_position)
            # if dxl_present_position < 2500:
            #     break

    def open_gripper(self):
        # while 1:
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_MX_GOAL_POSITION, self.open_pos)
            # dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_MX_PRESENT_POSITION)
            # print(dxl_present_position)
            # if dxl_present_position > 3100:
            #     break

# gripper = Gripper()
# gripper.open_gripper()
# time.sleep(2)
# gripper.close_gripper()
