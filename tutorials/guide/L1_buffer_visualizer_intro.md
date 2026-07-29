# **L1 Buffer Visualizer** 

L1 visualizer, [Tenstorrent Visualizer](https://github.com/tenstorrent/ttnn-visualizer), is a comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading. 

<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/l1_buffer_visualizer.png" width="60%" heigth="60%">
</div>
<br>

# **Installation guide**
## Prerequisite 

you need a Linux environment. 

## Installation

1. Go to your virtual env in the tt-metal folder
2. pip install ttnn-visualizer
3. (optional) I think you've already had id_rsa in .ssh directory. ssh-add to your agent.
You should add your ssh agent to notify which folder you're going use.

   ```cpp
    eval $(ssh-agent)
    ssh-add ~/.ssh/id_rsa

5. Run $ ttnn-visualizer and go to 127.0.0.0:8000


# How to Open Project
<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/howtoprojectopen-1.png" width="60%" heigth="60%">
</div>
<br>
<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/howtoprojectopen-2.png" width="60%" heigth="40%">
</div>
<br>

0. Before fetching l1 visualizer data from remote server, you should make some data.
Run your model with this environment variables.

```cpp
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_logging": true,
        "enable_graph_report": false,
        "report_name": "<name of the run in the visualizer>",
        "enable_detailed_buffer_report": true,
        "enable_detailed_tensor_report": false,
        "enable_comparison_mode": false
    }'
```
Then, directory "/to/your/tt-metal/generated/ttnn/reports" is created. You're going to fetch this.

## if you want remote access
1. First, click add_new_connection button.
2. Fill up the pop-up.
3. After making connections,click fetch folder list.
4. Folders are fetched, but the data aren't transferred. Click recycle button next to your folder name. That will fetch data inside the folder.
5. See your reports.
6. Your data is stored in ./virtual_env/lib/python3.10/site-packages/ttnn_visualizer/data/remote.
7. Those reports can be shown by clicking Select local report button.

## OR, download report folder from remote server, and run in localhost
1. $ ttnn-visualizer --profiler-path [DOWNLOADED GENERATED FOLDER]
2. go to 127.0.0.0:8000 and then just use.

# How to Use
In the menu at the top, you can observe the memory area through [Operations, Tensors, Buffers, Graphs], and you can usually check the L1 area for each operation by pressing [Memory details] after each Op in the [Operations] menu.

<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/howtouse-1.png" width="60%" heigth="60%">
</div>
<br>
<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/howtouse-2.png" width="60%" heigth="60%">
</div>
<br>
<div align="center">
<img src="https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/howtouse-3.png" width="60%" heigth="60%">
</div>
<br>

