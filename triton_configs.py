import triton
import triton.language as tl
def matmul_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4)
        ]


configs_any2D = [
    triton.Config(
        {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16},
        num_stages=2,
        num_warps=2,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32},
        num_stages=2,
        num_warps=2,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
        num_stages=2,
        num_warps=4,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32},
        num_stages=2,
        num_warps=4,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
        num_stages=2,
        num_warps=4,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
        num_stages=2,
        num_warps=4,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},
        num_stages=2,
        num_warps=4,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
        num_stages=2,
        num_warps=8,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256},
        num_stages=2,
        num_warps=8,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128},
        num_stages=2,
        num_warps=8,
    ),

    triton.Config(
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256},
        num_stages=2,
        num_warps=8,
    ),
]


# config_just_warps = [
#     triton.Config({}num_warps=1),
#     triton.Config(num_warps=1),
#     triton.Config(num_warps=2),
#     triton.Config(num_warps=4),
#     triton.Config(num_warps=4),
#     triton.Config(num_warps=8),
#     triton.Config(num_warps=8),
# ]