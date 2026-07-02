# Use official PyTorch image with CUDA 12.1 runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RCP CaaS requirement (Storage & UID mapping)
# Default values mapped to bendouro's EPFL/DCL credentials
ARG LDAP_USERNAME=bendouro
ARG LDAP_UID=274455
ARG LDAP_GROUPNAME=DCL-StaffU
ARG LDAP_GID=11260

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Set working directory to the user's home
WORKDIR /home/${LDAP_USERNAME}

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /home/${LDAP_USERNAME}/

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /home/${LDAP_USERNAME}

# Set owner of the files to the LDAP user
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Switch to the non-root user
USER ${LDAP_USERNAME}

# Default command
CMD ["bash"]
