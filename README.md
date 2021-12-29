# EdgeRec
This repository is the implement of EdgeRec using [Mobile Intelligent Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=109858)

### Prerequisites
  - A basic pytorch installation. The version is **1.7**.
  - Python packages you might not have: `jsonargparse`, `tqdm`, `sklearn`.

### Installation
1. Clone the repository
    ```Shell
    git clone https://github.com/tao-shen/EdgeRec
    ```
### Usage
1. Setup dataset

    We setup two datasize of data:  `full` refers to the full dataset and `demo` refers to the first 10000 samples of the full dataset. 
    
    We provide the demo dataset here and you can find the detailed description and download te full dataset [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=109858).
    ```Yaml
        # demo dataset
        datasize: demo
        device: cuda:0
        lr: 0.01
        batchsize: 100
        # full dataset
        datasize: full
        device: cuda:0
        lr: 0.001
        batchsize: 10000
    ```
2. Run the `main.py`, for example
    ```Shell
    python main.py --device=cuda:0 --datasize=demo --lr=0.01 --batchsize=100
    ```

## License
 

   
MIT License

Copyright (c) 2021 tao-shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.