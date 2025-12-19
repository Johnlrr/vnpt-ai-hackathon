# Base image CUDA theo yêu cầu
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Cài đặt Python & System deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3 -> python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Thiết lập thư mục làm việc là /code
WORKDIR /code

# Copy các file code vào /code
COPY requirements.txt .
COPY inference.sh .
COPY predict.py .

# Copy folder assets vào /code/assets
COPY assets /code/assets

# Cài đặt thư viện
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Cấp quyền thực thi
RUN chmod +x inference.sh

# Lệnh chạy mặc định
CMD ["bash", "inference.sh"]