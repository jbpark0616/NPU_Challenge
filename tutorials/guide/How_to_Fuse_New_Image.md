# How to Fuse New Booting Images

In the case that the board needs to boot up with new booting images, the current booting images on flash areas should be replaced with new booting images. USB **fastboot** is used for booting image update which is a tool within the android framework. To use the USB **fastboot**, a corresponding driver, called android driver, should be installed on host PC too.  
The steps below show how to install the android driver on host PC and how to download new booting binary images to the boards using USB **fastboot**.

<br>

## Step 1: Setting the configuration of **fastboot** on host PC with Ubuntu

1) Installing **fastboot**  
Download android platform-tools from [SDK Platform Tools release page](https://developer.android.com/tools/releases/platform-tools)

    <div align="center">

    <img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-android-platformtools-download.png" width="80%" heigth="80%">

     </div>

    Unzip the dowloaded archive to the proper location.( eg. `/home/bos/android/` )

    ```
    $ unzip platform-tools-latest-linux.zip
    Archive:  platform-tools-latest-linux.zip
      inflating: platform-tools/NOTICE.txt  
      inflating: platform-tools/adb      
      inflating: platform-tools/etc1tool  
      inflating: platform-tools/fastboot  
      inflating: platform-tools/hprof-conv  
      inflating: platform-tools/lib64/libc++.so  
      inflating: platform-tools/make_f2fs  
      inflating: platform-tools/make_f2fs_casefold  
      inflating: platform-tools/mke2fs   
      inflating: platform-tools/mke2fs.conf  
    extracting: platform-tools/source.properties  
      inflating: platform-tools/sqlite3 

    $ ls platform-tools
    lib64/      fastboot*            mke2fs*
    NOTICE.txt  hprof-conv*          mke2fs.conf
    adb*        make_f2fs*           source.properties
    etc1tool*   make_f2fs_casefold*  sqlite3*
    ```

    To make it easier to user, add the directory to the `PATH`
    ```
    $ export PATH=$PATH:/home/bos/android/platform-tools
    ```

    Make sure the version of the installed package is '35.0.2-12147458' or later.
    ```
    $ fastboot --version
    fastboot version 36.0.0-13206524
    Installed as /home/bos/android/platform-tools/fastboot
    ```
<br>

2) Adding device information by executing `fastboot_usb.sh` so that **fastboot** can recognize the USB device.  

    Setting necessary permissions for **fastboot** to use when the Eagle-N A0 board is connected as USB device.  

    Execute `fastboot_usb.sh`. While the script runs, the `root` privilege is required to create `/etc/udev/rules.d/51-android.rules` file. The script is covered.  

    ```fastboot_usb.sh
    $ cat fastboot_usb.sh 
    #!/bin/bash
    mkdir ~/.android
    cd ~/.android
    echo 0x3816 >> adb_usb.ini
    sudo wget -O /etc/udev/rules.d/51-android.rules https://raw.githubusercontent.com/NicolasBernaerts/ubuntu-scripts/master/android/51-android.rules
    echo -e "# BOS\nSUBSYSTEM==\"usb\", ATTR{idVendor}==\"3816\", MODE=\"0666\", GROUP=\"plugdev\"" | sudo tee -a /etc/udev/rules.d/51-android.rules
    sudo chmod a+r /etc/udev/rules.d/51-android.rules
    ```
<br>

3) Reboot the host PC.

<br>

## Step 2: Connecting Eagle-N A0 eval. board to host PC

*Figure 1* shows how to connect PC with Eagle-N A0 eval. board to download booting images using USB Type B cable for image download and USB Type C for console, respectively.  
**NOTE**: Confirm that the USB cables should support data transmission.( Do NOT use the cables that support only charging )

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-15.png" width="80%" heigth="80%">

Figure 1. Connecting PC and Eagle-N A0 eval. board using USB Type B & C

</div>

To download new booting binaries, **fastboot** packages should be installed on the host PC. The Eagle-N A0 eval. board should be connected to Eagle-N A0 eval. board first and Eagle-N A0 eval. board should be entered into **fastboot** mode during **uboot** phase.

<br>

## Step 2: Booting into USB **fastboot** mode.

1) Doing **power on reset** by turning on the power switch. Note that N1A0 eval. board should be connected to host PC before turning on the power switch.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-16.png" width="60%" heigth="60%">

Figure 2. Eagle-N A0 board power switch

</div>

<br>

2) Entering **uboot** command mode. Once you do **power on reset**, you can see the console logs as shown below and when you see the message `Hit any key to stop auto boot:`, press any key to enter the **uboot** command mode. Then you can see **uboot** command shell prompt, `u-boot=>`.
    ```
    Hit  any key to stop autoboot:   0
    u-boot=>
    u-boot=>
    ```

<br>

3) Configure eMMC/SD card partition by running `mmcpartwrite` command.
    ```
    u-boot=> mmcpartwrite
    ```
<br>

4) Run `fastboot` command as shown below. If you can see `crq->brequest:0x0` (sometimes not shown but that is ok), then Eagle-N A0 is successfully entered into **fastboot** mode. Now ready to download booting images. (The output message may change or may not be displayed, but if the USB driver is visible on the host PC, it is usable.)
    ```
    u-boot=>
    u-boot=> fastboot 0
    fastboot_buf_addr: 0x0000000095000000
    fastboot_buf_size: 0x0000000020000000
    serial# : EAGLE-N1-A0-FFFFFFFE
    ```
    Once you run `fastboot 0`, the device is ready for receiving booting images from host PC.  

<br>

5) Check if Eagle-N A0 is detected as USB device on the PC using “lsusb” command. If it is detected, `BOS Eagle N1 Bootloader Interface` should be displayed as shown below.
    ```
    $ lsusb
    Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
    Bus 001 Device 026: ID 10c4:ea70 Silicon Labs CP2105 Dual UART Bridge
    Bus 001 Device 023: ID 214b:7250 Huasheng Electronics USB2.0 HUB
    Bus 001 Device 010: ID 3816:1000 BOS Eagle N1 Bootloader Interface
    Bus 001 Device 009: ID 26ce:01a2 ASRock LED Controller
    Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
    ```
<br>

6) Check if `fastboot devices` are correctly detected
    ```
    $ fastboot devices
    EABLE-N1-A0-FFFFFFFE      Android Faastboot
    ```
<br>

7) If it doesn't work, check if each of the files, `adb_usb.ini` and `51-anroid.rules`, is properly created and contains the correct content.
    ```
    $ cat ~/.android/adb_usb.ini
    0x3816

    $ cat /etc/udev/rules.d/51-android.rules
    ...
    # BOS
    SUBSYSTEM=="usb", ATTR{idVendor}=="3816", MODE="0666", GROUP="plugdev"
    ```
<br>

## Step 3: Check the switches of Eagle-N A0

To switch the partition between eMMC and SD card, let choose the corresponding switch as below:

1. SW53:  
    a. 1, 2 LEFT: SD card  
    b. 1, 2 RIGHT: eMMC 

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-17.png" width="90%" heigth="90%">

Figure 3. SW 53

</div>

<br>

2. SW3  
    a. UP: eMMC  
    b. DONW: SD card  

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-18.png" width="90%" heigth="90%">

Figure 4. SW3

</div>

<br>

User can check the selected partition from **uboot** with `mmc info` command.  

- in case of using eMMC  
    ```
    u-boot=> mmc info
    Device: sdhci@20600000
    Manufacturer ID: 13
    OEM: 4e
    Name: AAM20E
    Bus SPeed: 52000000
    Mode: MMC High Speed (52MHz)
    Rd Block Len: 512
    MMC version 5.1
    High Capacity: Yes
    Capacity: 29.6 GiB
    Bus Width: 8-bit
    Erase Group Size: 512 KiB
    HC WP Group Size: 8 MiB
    User Capacity: 29.6 GiB WRREL
    Boot Capacity: 31.5 MiB ENH
    RPMB Capacity: 4 MiB ENH
    ```

- in case of using SD card  
    ```
    u-boot=> mmc info
    Device: sdhci@20600000
    Manufacturer ID: 1b
    OEM: 534d
    Name: EC1S5
    Bus SPeed: 52000000
    Mode: SD High Speed (50MHz)
    Rd Block Len: 512
    SD version 3.0
    High Capacity: Yes
    Capacity: 59.7 GiB
    Bus Width: 4-bit
    Erase Group Size: 512 Bytes
    ```

## Step 4: Fastboot download commands

After selecting the eMMC/SD card partition, the `fastboot flash` command is used to download the above images from host PC to the selected partition:  

You can check serial number with `fastboot devices` first, and download images to the eval. board.
```
$ fastboot devices
EAGLE-N1-A0-FFFFFFFE      Android Faastboot
```

```
$ fastboot -s EAGLE-N1-A0-FFFFFFFE flash package n1-bip.bin
Warning: skip copying package image avb footer (package partition size:0, package image size: 1604608)/
Sending 'package' (1567 KB)                    OKAY[   0.115s]
Writing 'package'                              OKAY[   5.842s]
Finished. Total time: 5.958s
```

If host PC is connected with only one Eagle-N A0 eval. board, you don't need to specify the serieal number
```
$ fastboot flash package n1-bip.bin
Warning: skip copying package image avb footer (package partition size:0, package image size: 1604608)/
Sending 'package' (1567 KB)                    OKAY[   2.735s]
Writing 'package'                              OKAY[   5.745s]
Finished. Total time: 8.484s
```

After flashing the image, to booting up the system into Linux, reboot the target board and enter the **uboot** download mode (refer the step 2-2).  
When the board is ready, download other images into the eMMC or SD card on the board:  

- Kernel devict tree blob (DTB)
- Kernel image
- Root file system


```
$ fastboot flash kerneldtb kernel.dtb
Sending 'kerneldtb' (36 KB)                    OKAY[   0.009s]
Writing 'kerneldtb'                            OKAY[   0.058s]
Finished. Total time: 0.303s

$ fastboot flash kernel kernel.img
Sending 'kernel' (240-9 KB)                    OKAY[   0.935s]
Writing 'kernel'                               OKAY[   2.031s]
Finished. Total time: 2.999s

$ fastboot flash rootfs rootfs_16.ext4
Sending 'rootfs' (16384 KB)                    OKAY[   1.101s]
Writing 'rootfs'                               OKAY[ 196.529s]
Finished. Total time: 197.665s

```

While downloading the images, you can see the progress from the board.  
For example,
```
u-boot=> fastboot 0
fastboot_buf_addr: 0x0000000095000000
fastboot_buf_size: 0x0000000020000000
serial# : EAGLE-N1-A0-FFFFFFFE
Starting download of 1604608 bytes
...........
downloading of 1604608 bytes finished
current version :0.0.0.1, package version: 0.0.0.1
active_slot: A

...
...


----- Partition update is complete!  ---
BIP version is identical, skipping update
Boot meta is identical, skipping update
New image flag set successfully
---------------------------------------
-- Package update was complete !  -----
---------------------------------------
```

<br>

The table below shows the list of binaries that need to be fused onto the board.

| Partition | Partition name | Image | Description |
| :---: | :---: | :---: | :--- |
| Package | package | n1-bip_[*release version*].bin | Total of image |
| Kernel DTB | kerneldtb | Kernel_[*release version*].dtb | The DTB of the Kernel |
| Kernel Images | kernel | Kernel_[*release version*].img |Linux Kernel Image |
| Root File System | rootfs | rootfs_[*release version*].img | Linux Root File System. sparse image of 512MB RootFS |

<div align="center"> Table 1. Partition information and image files </div>
