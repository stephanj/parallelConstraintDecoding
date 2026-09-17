# Parallel Constrained Decoding — web demo image (CPU inference via llama.cpp).
#
#   docker build -t pcd .
#   docker run --rm -p 8080:8080 pcd            # http://localhost:8080
#
# Layout of the build:
#   1. llama    — compiles libllama/libggml from the exact tag the Java FFM bindings were generated
#                 against (struct layouts must match). CPU backend only: on x86-64 one AVX2/FMA/F16C
#                 build (x86-64-v3, every cloud CPU has it); building all SIMD variants was too heavy
#                 for a hosted build instance.
#   2. model    — downloads the GGUF weights once (cached as their own layers).
#   3. build    — mvn package.
#   4. runtime  — slim JRE + the libs + the jar + presets + model.

ARG LLAMA_TAG=v0.4.1
# Models baked into the image: the ~1B class of Qwen 2.5 and Qwen 3.5. GGUF_FILE is the default one.
ARG GGUF_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
ARG GGUF_FILE=qwen2.5-1.5b-instruct-q8_0.gguf
ARG GGUF2_URL=https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
ARG GGUF2_FILE=Qwen3.5-0.8B-Q8_0.gguf

# ---------------------------------------------------------------- 1. llama.cpp
FROM debian:bookworm-slim AS llama
ARG LLAMA_TAG
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp.git /src
# x86-64 (cloud hosts): AVX2 + FMA + F16C + BMI2 (x86-64-v3).
# arm64 (local test on Apple Silicon Docker): one portable ARMv8.2 build with dotprod (i8mm is not exposed in Docker VMs).
# -w: llama.cpp emits thousands of harmless warnings that can overflow a hosted build log.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        CPU_FLAGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON"; \
    else \
        CPU_FLAGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; \
    fi \
    && JOBS=$(nproc); [ "$JOBS" -gt 8 ] && JOBS=8; \
    cmake -S /src -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_C_FLAGS=-w -DCMAKE_CXX_FLAGS=-w \
        -DGGML_NATIVE=OFF ${CPU_FLAGS} \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build /src/build --config Release --target llama ggml-cpu -j"$JOBS" \
    && mkdir -p /out/lib \
    && find /src/build -name 'libllama*.so*' -exec cp -a {} /out/lib/ \; \
    && find /src/build -name 'libggml*.so*' -exec cp -a {} /out/lib/ \; \
    && ls -la /out/lib

# ---------------------------------------------------------------- 2. model
FROM debian:bookworm-slim AS model
ARG GGUF_URL
ARG GGUF_FILE
ARG GGUF2_URL
ARG GGUF2_FILE
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models && curl -fL --retry 5 -o /models/${GGUF_FILE} "${GGUF_URL}"
RUN curl -fL --retry 5 -o /models/${GGUF2_FILE} "${GGUF2_URL}"

# ---------------------------------------------------------------- 3. java build
FROM eclipse-temurin:25-jdk AS build
RUN apt-get update && apt-get install -y --no-install-recommends maven && rm -rf /var/lib/apt/lists/*
WORKDIR /app/java
COPY java/pom.xml .
RUN mvn -q -B dependency:go-offline
COPY java/src ./src
RUN mvn -q -B clean package -DskipTests

# ---------------------------------------------------------------- 4. runtime
FROM eclipse-temurin:25-jre
ARG GGUF_FILE
# libgomp: the CPU backend of ggml is built with OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PCD_LLAMA_LIB_DIR=/opt/llama/lib \
    LD_LIBRARY_PATH=/opt/llama/lib \
    GGML_BACKEND_PATH=/opt/llama/lib \
    PCD_GGUF=/app/models/${GGUF_FILE} \
    PCD_BIND=0.0.0.0 \
    PORT=8080 \
    PCD_READ_ONLY=true \
    PCD_SEQUENTIAL_RACE=true \
    JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED"
WORKDIR /app
COPY --from=llama /out/lib /opt/llama/lib
COPY --from=model /models /app/models
COPY --from=build /app/java/target/pcd-benchmark.jar /app/pcd-benchmark.jar
COPY presets /app/presets
RUN mkdir -p /app/results
EXPOSE 8080
CMD ["java", "-jar", "/app/pcd-benchmark.jar", "serve"]
