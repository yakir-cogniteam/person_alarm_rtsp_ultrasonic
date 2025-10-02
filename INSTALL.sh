# sudo apt install python3-opencv
# pip3 install onvif-zeep --break-system-packages
# cd /home/pi/.local/lib/python3.11/site-packages/wsdl/
# cd /home/pi/.local/lib/python3.11/site-packages/wsdl
# cd /home/pi/.local/lib/python3.11/site-packages/
# mkdir -p wsdl
# cd wsdl/
# wget https://www.onvif.org/ver10/device/wsdl/devicemgmt.wsdl
# wget https://www.onvif.org/ver10/media/wsdl/media.wsdl
# wget https://www.onvif.org/ver20/ptz/wsdl/ptz.wsdl
# wget https://www.onvif.org/ver10/events/wsdl/event.wsdl
# wget https://www.onvif.org/ver20/imaging/wsdl/imaging.wsdl

# cd /home/pi/.local/lib/
# mkdir -p ver10/schema
# mkdir -p ver20/schema
# wget https://www.onvif.org/ver10/schema/onvif.xsd
# wget https://www.onvif.org/ver10/schema/common.xsd
# cd ../../ver20/schema
# wget https://www.onvif.org/ver20/schema/onvif.xsd

pip3 install pytapo==3.3.15 python-kasa==0.5.0
