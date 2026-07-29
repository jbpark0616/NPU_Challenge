#!/bin/bash
mkdir ~/.android
cd ~/.android
echo 0x3816 >> adb_usb.ini
sudo wget -O /etc/udev/rules.d/51-android.rules https://raw.githubusercontent.com/NicolasBernaerts/ubuntu-scripts/master/android/51-android.rules
echo -e "# BOS\nSUBSYSTEM==\"usb\", ATTR{idVendor}==\"3816\", MODE=\"0666\", GROUP=\"plugdev\"" | sudo tee -a /etc/udev/rules.d/51-android.rules
sudo chmod a+r /etc/udev/rules.d/51-android.rules

