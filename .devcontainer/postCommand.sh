#!/bin/bash

dnf install -y http://10.30.38.131/kojifiles/work/tasks/748/2600748/python-safetensors-0.4.5-1.uos25.1.x86_64.rpm
dnf install -y mindspore
dnf install -y wget unzip 

# link python3 to python
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

python -m pip install --root-user-action ignore --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install --root-user-action ignore matplotlib download pyarrow datasets