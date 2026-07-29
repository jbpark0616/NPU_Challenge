# **Evaluation Board Overview**

## Eagle-N A0 Evaluation Board Specification

### Board Overview

*Figure 1*  and *Figure 2* show the top view and side view of the Eagle-N A0 evaluation board, respectively.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-01.jpg" width="50%" height="50%" title="top view"/> 

Figure 1. Top view of Eagle-N A0 evaluation board

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-02.jpg" width="50%" height="50%" title="top view"/> 

Figure 2. Side view of Eagle-N A0 evaluation board

</div>

<br>

### Physical dimensions

The dimensions of PCB and mechanical frame are shown in *Figure 3* and *Figure 4* respectively.

<div align="center">

| | PCB | EVB Mechanical Frame Specification |
| :---: | :---: | :---: |
| Height | 144 mm   | 144 mm |
| Length | 268 mm  | 268 mm  |
| Thickness | 1.59 mm (18 layers) | 49mm |

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-03.jpg" width="50%" heigth="50%">

Figure 3. Bare board dimension

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-04.png" width="50%" heigth="50%">

Figure 4. Mechanical frame dimension

</div>

<br>

### Power requirements

- Input voltage: 100 ~ 240 V - 6.0 A
- Frequency: 50/60 Hz
- Output voltage: 12 V, 20 A

<br>

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-05.jpg" width="20%" heigth="20%">

Figure 5. Power adapter

</div>

### Connectivity information
*Figure 6* shows the connector locations on the evaluation board and *Table 1* describes each of the numbered components according to the numbers marked on *Figure 6*.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-06.png" width="40%" heigth="40%">

Figure 6. Connectivity information

<br>

| No | Description |
| :--- | :--- |
| 1 | Connector Male type 12V / 20A over |
| 2 | Power switch On/Off |
| 3 | USB2.0 Interface connector |
| 4 | UART connector for CA53, CM7 |
| 5 | Trace32 / JTAG interface. For JTAG, only 20pin Connector |
| 6 | High Speed Connector for Quad/Octal SPI. (NOR Flash, MicroSD, eMMC, etc.) |
| 7 | PCIe Gen5 Edged Card interface |

Table 1. Connector Descriptions

</div>

<br>

### Default Switch Settings
*Figure 7* shows the switch locations on the evaluation board. Each of the numbered components in the diagram is linked to the corresponding tables below. The table includes the reference number. In case the board does not work for some reason, all default switch settings should be confirmed first.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-07.png" width="80%" heigth="80%">

Figure 7. Switch locations on the evaluation board

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-table-02.png
" width="90%" heigth="90%">

Table 2. CPU Default Slide Switch Settings

<br>

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-table-03.png" width="90%" heigth="90%">

Table 3. MSHC Default Slide Switch Settings

</div>

<br>

### Support of Operation modes
The operation mode is decided by setting OPMODE pin on the board. Based on OPMODE pin setting, the booting source is selected and booting images are loaded from the selected source. For Eagle-N A0 eval. board, the default value is `OPMODE3 = 0` meaning that QSPI flash device is the selected booting source.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-table-04.png" width="90%" heigth="90%">

Table 4/ Operation Mode (OM) Information
</div>

<br>

## Eagle-N A0 Memory Map

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-08.png" width="70%" heigth="70%">

Figure 8. Eagle-N A0 memory map

</div>

<br>

## NPU Block diagram
*Figure 9* shows the NPU block diagram within Eagle-N A0. There are APB and AXI interfaces. For the AXI interface, master and slave ports exist.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-09.png" width="90%" heigth="90%">

Figure 9. NPU block diagram

</div>

<br>

## Eagle-N A0 SW Overview
Eagle-N A0 software components reside in both Eagle-N A0 and host AP as shown in *Figure 10*. Eagle-N A0 and the host AP are interconnected via PCIE Gen5 interface. The connection between Eagle-N A0 and the host AP is established in uboot during booting-up phase. The entire booting sequence of Eagle-N A0 can be found in *Figure 10*.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-10.png" width="90%" heigth="90%">

Figure 10. SW components on RTCore, Cortex-M7, Cortex-A53, and host AP

</div>

<br>

## Eagle-N A0 Booting Sequence
This section describes the booting sequence till Linux OS runs. This sequence shows what software components are loaded and their role for the next step. Note that in this release version, Linux kernel binary is not included.

<div align="center">

<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/app-note-figure-booting-sequence.jpg" width="90%" heigth="90%">

Figure 11. N1 Booting sequence

</div>



