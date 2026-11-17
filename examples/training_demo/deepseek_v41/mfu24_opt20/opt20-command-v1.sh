set -eo pipefail
root=/xw-tree/full-v41
stage="$root/full-model/S1"
printf '%s  %s
' 2e542f80a0b6b169a59ee73d888e1d8a83272f78b51cb43c8cc478cde6ba8b22 "$root/2e542f80-launch-full-portable.sh" 020bf2eb1abf2e3832bc900805c010ff171bd2394c0b0854db737f38b4e66be2 "$stage/020bf2eb-train-full-v41-nondeterministic.py" 39b992335a046a8b2ae5d0f46eefb870e542f4aae375d3d9b0908f90798c89e2 "$stage/39b99233-profile-backbone-stages.py" | sha256sum -c -
printf '%s  %s
' b15c7b6004fd5db421283d04ceabf75e352923eb4acfedede4be5fd1a7221626 "$root/runtime-cann921/install/runtime-cann921-env-b15c7b60.sh" | sha256sum -c -
source "$root/runtime-cann921/install/runtime-cann921-env-b15c7b60.sh"
export DSV41_FULL_REPO="$root/hp-backbone-cann921-eec395c4d5b1456fa0338c3beac8cbf222b41fca-lf"
printf '%s  %s
' 2104ecd1f61a6fadf887dd570d6fe85a551a6d5da3bf703ae8d2fa3e5955df75 "$DSV41_FULL_REPO/hyper_parallel/components/functional/_triton/v41_partial_rope.py" | sha256sum -c -
export DSV41_FULL_WORKER="$root/51999d36-bootstrap-repaired-affinity.py" DSV41_FULL_MODE=benchmark
export DSV41_FULL_DIAGNOSTICS="$root/opt20-observe-vec-rope-skip1-clip-muon.py" DSV41_FULL_EXPECT_FUSED_KL=1
printf '%s  %s
' 08fbcd99d2a537b9cda6d645144b07ca6b395837df08320bdd97dccd0c364612 "$DSV41_FULL_DIAGNOSTICS" | sha256sum -c -
export HP_EP_EQUAL_A2A=1 PYTHONFAULTHANDLER=1
export DSV41_CSA_PERFORMANCE_PARENT_SHA=bb572c9ab5ed1eeacc16c6d9be36a04c164f20770a59c7990896eb55e13dce2b
export DSV41_SHARD_PARENT="/xw-tree/full-v41/0fe83cff-observe-dspark-local-checkpoint.py"
export DSV41_SHARD_PARENT_SHA=0fe83cff302ca0b90820b8ada68970757b648118ddff520779ea83f713fa466b DSV41_EXPECT_DENSE_SHARDS=32
printf '%s  %s
' 51999d36b35c41fcf70cde6c887f846dffee39fff7e4518ee06d5fded0db8b1d "$root/51999d36-bootstrap-repaired-affinity.py" | sha256sum -c -
exec bash "$root/2e542f80-launch-full-portable.sh" huangxs-opt20-vec-rope-skip1-clip-muon-repeat30-v1 eec395c4d5b1456fa0338c3beac8cbf222b41fca dsv41-vlm-256die-16k-contained-recompute-20c1e5cd-repeat30-v1 30
