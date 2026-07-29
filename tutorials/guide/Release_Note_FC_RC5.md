# Eagle-N A0 FC-RC5 Release Note 

- Host SW Release Version: v0.1.0.5
- Matching device SW version: v0.1.0.3_patch1119
- Release Date: December 4th, 2025

<br>

This package includes docker-based model execution and software installation as same as previous release. There is no change in Eagle-N evaluation board software stack, so the binary from the FC-RC4 release can be used for the Eagle-N evaluation board.

<br>

## New features
- E2E performance optimized version of FastOFT and Deeplab V3 Plus (a variant of Panoptic-Deeplab)
    - To run the FastOFT demo, download checkpoint dataset from [here](https://d250qm1zxxsfut.cloudfront.net/EagleN-FC-RC5/nano_4m_oft8_tp8x128_nearest_dr_06_no_dist-0600.pth.gz?Expires=1767229477&Signature=pt1j2DBX7jEZaNofuMKjXe4GYSgf1Q7~PjORkpL0~yQtrKaz5DmutmVyzx0h5oVaKcSzSyH9pZ-tZViOTH-bLklxnbmFFjs7DbyxtJXfzNoJxzWwIc4RncKz2CCgMFiA9ouqc7OYVtRgrwn3y17PEXSCVfMmL0bN7W-9ZKA0o2cdzlyuUFY5b5lqABf3LblvWTJ5exi7iXMUh7jgcilWZJmVQpt8kkXS1oT7EFqvKYTvskzu6FaEnxm5YSs4ANWxjgmEN0-DFFvv7RFuaj-YMyFQknrWEOzYyq98VklrT1~SCi0bvGn9HZCvkKtgkd9L7vgCwCG0lJ75UlxzXAu2GA__&Key-Pair-Id=KMRE43UBHN9IJ)
- L1 visualization and Tracy profiling support for all the supported models

<br>

## Supported AI Models and Their Performance

### LLM and VLM

| Model | Batch size| ttft (time to first token) | t/s/u | t/s | DRAM / PCIe |
| ---: | :---: | :---: | :---: | :---: | :---: |
| QWEN 2.5VL 3B | 2 | 1183.11 ms | 9.51 | 19.01 | LPDDR5x 7.5GHz / Gen5 |
| QWEN 2.5VL 7B| 2 | 1422.01 ms | 4.36 | 8.71 | LPDDR5x 7.5GHz / Gen5 |
| LLaMA 3.1 8B | 16 |  232.63 ms | 4.82 | 77.17 | LPDDR5x 7.5GHz / Gen5 |
| LLaMA 3.2 1B | 16 | 51.4 ms | 22.71 | 363.36 | LPDDR5x 7.5GHz / Gen5 |
| LLaMA 3.2 3B | 16 | 100.3 ms | 9.97 | 159.53 | LPDDR5x 7.5GHz / Gen5 |

<br> 

### Object detection

| Model | Batch size | Input size | FPS | DRAM / PCIe |
| :---: | :---: | :---: | :---: | :---: |
| SSR | 1 | [ 640 x 384 ] x 6 | 4.61 | LPDDR5x 7.5GHz / Gen5 |
| YOLOv8s | 1 | 256 x 256 | 164 | LPDDR5x 7.5GHz / Gen5 |
| | 1 | 320 x 320  | 138 | LPDDR5x 7.5GHz / Gen5 |
| DeepLab V3+ | 1 | 512 x 1024 | 18.03 | LPDDR5x 7.5GHz / Gen5 |
| PDL | 1 | 512 x 1024 | 12.24 | LPDDR5x 7.5GHz / Gen5 |
| FastOFT | 1 | 384 x 1280| 16.13 | LPDDR5x 7.5GHz / Gen5 |
| ResNet50 | 4 | 224 x 224 | 976 | LPDDR5x 7.5GHz / Gen5 |

<br>

## List of Deliverables

### Documents

* [Evaluation_Board_Overview.md](./Evaluation_Board_Overview.md)
* [How_to_Setup_Board.md](./How_to_Setup_Board.md)
* [How_to_Install_SW_Stack.md](./How_to_Install_SW_Stack.md)
* [How_to_Fuse_New_Image.md](./How_to_Fuse_New_Image.md)
* [Using_Host_Display_inside_Docker.md](./Using_Host_Display_inside_Docker.md)
* [Tracy_intro.md](./Tracy_intro.md) 
* [L1 Visualizer_intro.md](./L1_buffer_visualizer_intro.md)
  
### Source code

* [BOS UMD v0.1.0.5](https://github.com/bos-semi-release/BOS_UMD/tree/v0.1.0.5) 
* [BOS KMD v0.1.0.5](https://github.com/bos-semi-release/BOS_KMD/tree/v0.1.0.5)
* [tt-metal v0.1.0.5](https://github.com/bos-semi-release/tt-metal/tree/v0.1.0.5)
* [tt-llk v0.1.0.5](https://github.com/bos-semi-release/tt-llk/tree/v0.1.0.5)
* [BOS Tools v0.1.0.5](https://github.com/bos-semi-release/BOS_TOOLS/tree/v0.1.0.5)

### Docker image and Dockerfile

Refer to [How to Use Docker Container](./How_to_Use_Docker_Container.md) when you want to use Docker conatainer for development enviroment.

* [fc_rc5.7z](https://d2ycbxqnpx761l.cloudfront.net/fc_rc5/fc_rc5_final.7z?Expires=1767438840&Signature=zzarj0lXPV7YshJat5hpWfld47nwR9NYfRDdRxtLppVq-xyuSDeZkRHZEg9d86tNrL7Pr0UUKo3IRmGiR6yeB3EYMdnygceakN7b-jigSrRK9i5Bwz8DPourMNUDo9KK9LTXCmfbC1XDXx3xCILA0g9U0IZvJgH0zh1N2UI3nWpImokbPBXfcQD7PX-jVokDavKqRkWhx3viXLtNkLapuGN2UGauviNXtSQ-4iD1kvpN9v931oHqiyERtbEq0eNWEqOy2QqfE8sXt6wauQfoarBNyMnEhl2GU1HkAVg2wyV8T2UUtWFsnucf3abkDs43iTMHT-4RzcUXNUw6TjW0HA__&Key-Pair-Id=KMRE43UBHN9IJ) (This is valid until December 31th, 2025 )  
* [Dockerfile](./docker/Dockerfile)   
 
### Binary images from FC-RC4 release

* [Package](./binaries/n1-bip-v0.1.0.3_patch1119.bin)  
* [Kernel device tree](./binaries/kernel-v0.1.0.3_patch1119.dtb)  
* [Kernel](./binaries/kernel-v0.1.0.3_patch1119.img)  
* [Root FS](./binaries/rootfs-v0.1.0.3_patch1119.img)


## Known Issues

- ViT model under the bos_model directory has not been verified.
