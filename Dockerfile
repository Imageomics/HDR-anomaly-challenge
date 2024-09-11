FROM codalab/default-cpu

RUN apt update && apt upgrade -y
RUN apt install python3.10 -y
RUN apt install python3.10-venv -y
RUN python3.10 -m ensurepip --upgrade
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN python3.10 -m pip install --upgrade pip
RUN pip install pillow tqdm pandas scikit-learn pyyaml
