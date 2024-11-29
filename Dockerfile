FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y build-essential python3-full python3-venv
RUN python3 -m venv /root/cotengra_python && \
    /root/cotengra_python/bin/pip install cotengra && \
    echo 'source /root/cotengra_python/bin/activate' >> /root/.bashrc
ENV VIRTUAL_ENV="/root/cotengra_python" \
    LD_LIBRARY_PATH="$/root/cotengra_python/lib:/usr/lib/python3.12/config-3.12-x86_64-linux-gnu:${LD_LIBRARY_PATH}" 
ENV PATH="/root/cotengra_python/bin/:${PATH}"
ENV DEBIAN_FRONTEND="noninteractive" \
    PATH="/root/.cargo/bin:${PATH}"
COPY rust-toolchain.toml .
# 1. Install dependencies for the toolchain
RUN apt-get install -y openssh-client jq && \
    # 2. Install rust
    apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --no-modify-path && \
    # 3. Trigger installation of rust
    rustup show && \
    # 4. Clear intermediate files
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock tmp/
ARG SSH_PRIVATE_KEY
RUN --mount=type=ssh \
    # 1. Install dependency crates
    # 1.1 Create a dummy project
    cargo new dummy && \
    mv -f tmp/Cargo.toml tmp/Cargo.lock rust-toolchain.toml dummy && \
    cd dummy && \
    # 1.2 Need to create a dummy benchmark directory as well
    mkdir benches && touch benches/benchmarks.rs && \
    # 1.3 Set up SSH access to Gitlab
    mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan -H gitlab.lrz.de >> ~/.ssh/known_hosts && \
    if [ -z "${SSH_PRIVATE_KEY}" ]; then \
        # To use SSH in local builds, use `docker build --ssh default .`
        # (or `docker build --ssh default=${HOME}/.ssh/id_rsa .`} for a specific key)
        # If the key is password protected, you can use `ssh-add ${HOME}/.ssh/mykey` first
        echo "No SSH_PRIVATE_KEY provided, using SSH from the host"; \
    else \
        # The SSH_PRIVATE_KEY build argument must be a base64-encoded private key
        # See https://stackoverflow.com/a/38570269
        # and https://www.programonaut.com/how-to-mask-an-ssh-private-key-in-gitlab-ci/
        echo "The SSH_PRIVATE_KEY variable was provided"; \
        eval $(ssh-agent -s); \
        if [ "${SSH_PRIVATE_KEY}" = "default" ]; then \
            echo "SSH_PRIVATE_KEY is set to default value, no private repositories will be accessible"; \
        else \
            echo "Adding SSH_PRIVATE_KEY"; \
            echo "$SSH_PRIVATE_KEY" | base64 -d | tr -d '\r' | ssh-add -; \
        fi; \
    fi && \
    # 1.4 Install dependencies
    cargo fetch && \
    # 2. Clean intermediate files
    cd .. && \
    rm -rf dummy