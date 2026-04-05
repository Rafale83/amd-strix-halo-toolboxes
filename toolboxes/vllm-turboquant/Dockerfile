# vLLM + TurboQuant KV Cache Compression for AMD Strix Halo (gfx1151)
# Layer on top of kyuz0's vLLM container — adds TurboQuant plugin
# TurboQuant: https://github.com/Alberto-Codes/turboquant-vllm
#
# Build:
#   podman build -t vllm-turboquant-gfx1151 -f Dockerfile .
#
# Usage:
#   toolbox enter vllm-turboquant
#   start-vllm --model <model> --attention-backend CUSTOM
#
# TurboQuant compresses KV cache from FP16 to 3-4 bits (~3.7x compression)
# allowing much larger context windows on the same memory budget.
# It does NOT quantize model weights.

FROM docker.io/kyuz0/vllm-therock-gfx1151:latest

LABEL maintainer="Rafale83"
LABEL description="vLLM + TurboQuant KV cache compression for Strix Halo gfx1151"

# Install TurboQuant plugin (without [vllm] extra to preserve the patched ROCm vLLM)
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps turboquant-vllm && \
    /opt/venv/bin/pip install --no-cache-dir molmo-utils einops

# TurboQuant environment defaults
# TQ4_K_BITS/TQ4_V_BITS: bits for Key/Value quantization (3 or 4)
ENV TQ4_K_BITS=4
ENV TQ4_V_BITS=4

# Convenience wrapper that sets --attention-backend CUSTOM by default
COPY start-vllm-tq /opt/start-vllm-tq
RUN chmod +x /opt/start-vllm-tq && \
    ln -sf /opt/start-vllm-tq /usr/local/bin/start-vllm-tq

# Banner update
RUN printf '\n# TurboQuant KV Cache Compression\n\
echo "  TurboQuant: KV cache ${TQ4_K_BITS}/${TQ4_V_BITS}-bit compression active"\n\
echo "  Use: start-vllm-tq --model <model>  (auto-sets --attention-backend CUSTOM)"\n\
echo "  Or:  start-vllm --model <model> --attention-backend CUSTOM"\n\
echo "  Env: TQ4_K_BITS=${TQ4_K_BITS} TQ4_V_BITS=${TQ4_V_BITS}"\n\
echo ""\n' >> /etc/profile.d/99-toolbox-banner.sh

CMD ["/bin/bash"]
