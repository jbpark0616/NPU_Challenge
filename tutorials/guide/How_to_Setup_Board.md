# **How to Set up Environment**


## Eagle-N A0 Eval. Board Setup Environment to run AI models

Due to the form factor, the Eagle-N A0 Eval. board may not be directly inserted into PCIe Slot on the motherboard of host PC. A PCIe riser cable in the Eagle-N A0 eval. The board package should be used to connect Eagle-N A0 board and host PC as shown in *Figure 1*.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-12.jpg" width="50%" heigth="50%">

Figure 1. Host PC – Eagle-N A0 eval board connection via PCIe riser cable

<br>

### Host PC and Eagle-N A0 System Requirements

| Host PC | Eagle-N A0 Eval board |
| :--- | :--- |
| Intel i7 | CPU: ARMv8 Cortex-A53, 1.3Ghz |
| Ubuntu 22.04 LTS | Kernel: Linux 6.1.15 |
| PCIe Gen1 to Gen5 | Bootloader: uboot 23.4 |

</div>

<br>

### Host PC BIOS Setting
Since the N1 board uses the clock provided by the host, the boot sequence between the host PC and the N1 board must be synchronized. Therefore, the N1 board is designed to wake the host PC through a wakeup signal from the PCIe endpoint. As a result, the BIOS settings on the host PC need to be modified
according to the values shown in the two tables below. Note that at present, the GIGABYTE BIOS is the only configuration that has been officially validated for use as a host PC. Although other BIOS types may operate correctly when configured in a similar manner, their compatibility with the current N1 software version has not been verified yet.

<div align="center">

| Item | Setting value |
| :--- | :--- |
| Wake on PCIe / Wake from PCIe | Enable |
| PME Event Wake Up| Enable |
| S5 Wake on PME | Enable |

Table 1. BIOS settings related to PCIe (may not be relevant depending on BIOS version)

</div>

<br> 

If the evaluation board DRAM BAR size is 512MB (default: 128MB), check and confirm the host BIOS setting to support board is the same as following table. In the case of 128MB, the BAR resizing capability should be enabled, and the rest doesn’t matter. *Figure 2* shows the GIGABYTE BIOS setup screenshot.

<div align="center">

| Configured BAR size | 512 MB | 128MB ( default ) |
| ---- | ---- | ---- |
| Above 4G Decoding | Disabled | Doesn’t matter |
| Above 4G MMIO BIOS assignment | Disabled | Doesn’t matter |
| Re-Size BAR Support | Disabled | Enabled |

Table 2. BIOS settings related to DRAM BAR size

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-13.jpg" width="90%" heigth="90%">

Figure 2. Host PC – BIOS configuration settings for GIGABYTE

</div>

`lspci -v` shows the evaluation board information when the board and Host PC are successfully connected.

* In the case of a 128MB BAR size 
    ```
    $ lspci -v
    01:00.0 Co-processor: Synopsys, Inc. DWC_usb3 / PCIe bridge
        Flags: bus master, fast devsel, latency 0, IRQ 156
        Memory at a8000000 (32-bit, prefetchable) [size=128M]
        Memory at a0000000 (32-bit, prefetchable) [size=128M]
        Memory at b0000000 (32-bit, prefetchable) [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: bos
        Kernel modules: bos
    ```

* In the case of a 512MB BAR size 
    ```
    $ lspci -v
    01:00.0 Co-processor: Synopsys, Inc. DWC_usb3 / PCIe bridge
        Flags: bus master, fast devsel, latency 0, IRQ 156
        Memory at 60000000 (64-bit, prefetchable) [size=512M]
        Memory at 48000000 (64-bit, non-prefetchable) [size=128M]
        Memory at 50000000 (32-bit, non-prefetchable) [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: bos
        Kernel modules: bos
    ```

The default BAR size of Eagle-N A0 evaluation board is **128MB**

<br>

## How to Boot the Eval. board

There are four steps to get the N1 eval. board started with host PC.

1. Connect a host PC and N1 Eval Board using PCIe cable and Connect USB-to-type C cables to see console logs. There are two type C and one microUSB (a.k.a type B) ports on the board as *figure 1*.  
Each of the cables supports two ports, so there will be 4 serial ports as shown below once you connect the cables to host PC and Eagle-N A0 board. The console terminal can be opened to each of the serial parts shown below.
    ```
    $ ll /dev/ttyUSB*
    crw-rw---- 1 root dialout 188, 0 Nov  3 17:42 /dev/ttyUSB0
    crw-rw---- 1 root dialout 188, 1 Nov  3 17:42 /dev/ttyUSB1
    crw-rw---- 1 root dialout 188, 2 Nov  3 17:42 /dev/ttyUSB2
    crw-rw---- 1 root dialout 188, 3 Nov  3 17:42 /dev/ttyUSB3
    ```
<br>

2. Eagle-N A0 eval. board booting.  
Initially, booting images are fused on NOR flash and eMMC (as default) on the eval. board when the board is shipped. So once the board is powered on, the booting process proceeds automatically and once it is done, the system will boot directly into the Linux kernel. The serial port, `/dev/ttyUSB0`, below could vary. To log into the Linux Kernel, use below account and password:  
    - username: `root`  
    - password: `bossemi`.
    ```
    Poky (Yocto Project Reference Distro) 4.0.21 n1bos /dev/ttyS0

    n1bos login: root
    password:

    root@n1bos:~#
    ```
<br>

3. If any key is hit during the Uboot and return to the Uboot promt, to boot from Uboot to Linux kernel, use below command:  
    ```
    Hit any key to stop autoboot:   0
    u-boot =>
    u-boot => run bootcmd_bootwithmode
    ```
<br>

### Checking PCIe link connection status
Run `lspci -v` command from the terminal in the host PC, then “Synopsis, Inc. DWC_usb3..” should be found from the output results shown below.

* In the case of a 128MB BAR size 
    ```
    $ lspci -v
    01:00.0 Co-processor: Synopsys, Inc. DWC_usb3 / PCIe bridge
        Flags: bus master, fast devsel, latency 0, IRQ 156
        Memory at a8000000 (32-bit, prefetchable) [size=128M]
        Memory at a0000000 (32-bit, prefetchable) [size=128M]
        Memory at b0000000 (32-bit, prefetchable) [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: bos
        Kernel modules: bos
    ```

* In the case of a 512MB BAR size 
    ```
    $ lspci -v
    01:00.0 Co-processor: Synopsys, Inc. DWC_usb3 / PCIe bridge
        Flags: bus master, fast devsel, latency 0, IRQ 156
        Memory at 60000000 (64-bit, prefetchable) [size=512M]
        Memory at 48000000 (64-bit, non-prefetchable) [size=128M]
        Memory at 50000000 (32-bit, non-prefetchable) [size=2M]
        Capabilities: <access denied>
        Kernel driver in use: bos
        Kernel modules: bos
    ```
  
