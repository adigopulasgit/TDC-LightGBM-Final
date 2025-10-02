FROM continuumio/miniconda3

WORKDIR /app

# Install system dependencies for tiledb/rdkit
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy env and install
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate tdc_lgbm" >> ~/.bashrc

SHELL ["conda", "run", "-n", "tdc_lgbm", "/bin/bash", "-c"]

COPY src/ src/
COPY notebooks/ notebooks/
COPY configs/ configs/
COPY results/ results/
COPY models/ models/
COPY requirements.txt .
COPY README.md .

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
