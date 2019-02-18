

from bluepy import btle

class ScanDelegate(btle.DefaultDelegate):
    def __init__(self):
        btle.DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print ("Discovered device"+ dev.addr)
        elif isNewData:
            print ("Received new data from"+ dev.addr)

scanner = btle.Scanner().withDelegate(ScanDelegate())
devices = scanner.scan(10.0)

for dev in devices:
    print ("Device %s (%s), RSSI=%d dB" % (dev.addr, dev.addrType, dev.rssi))
    for (adtype, desc, value) in dev.getScanData():
        print ("  %s = %s" % (desc, value))


import binascii
import struct
import time
 
temp_uuid = btle.UUID(0x2221)
 
p = btle.Peripheral("C0:DB:DF:10:BD:A4", "random")
 
try:
    ch = p.getCharacteristics(uuid=temp_uuid)[0]
    if (ch.supportsRead()):
        while 1:
            val = binascii.b2a_hex(ch.read())
            val = binascii.unhexlify(val)
            val = struct.unpack('f', val)[0]
            print (str(val) + " deg C")
            time.sleep(1)
 
finally:
    p.disconnect()