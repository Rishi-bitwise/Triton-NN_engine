from triton_configs import matmul_autotune_config, configs_any2D #, config_just_warps

import triton
import torch
import triton.language as tl


@triton.autotune(
    configs=matmul_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tmatmul(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr
            ):
            
            pid = tl.program_id(axis=0)

            total_block_m = tl.cdiv(M, BLOCK_SIZE_M)
            total_block_n = tl.cdiv(N, BLOCK_SIZE_N)
            total_block_k = tl.cdiv(K, BLOCK_SIZE_K)

            total_blocks_per_group = GROUP_SIZE * total_block_n

            group_idx = pid // total_blocks_per_group
            group_start = group_idx * GROUP_SIZE

            block_idx_in_group = pid % total_blocks_per_group

            new_group_size = min(total_block_m-group_start, GROUP_SIZE)

            block_m_off_in_group = block_idx_in_group % new_group_size
            block_n_off_in_group = block_idx_in_group // new_group_size

            pid_m = group_idx*GROUP_SIZE + block_m_off_in_group
            pid_n = block_n_off_in_group

            #Now do blocked Matmul

            offs_am = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)           #rows of A pointers
            offs_bn = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)           #cols of B pointers

            offs_k = tl.arange(0, BLOCK_SIZE_K)

            A_ptrs = A_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak)
            B_ptrs = B_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn)
            
            accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(total_block_k):
                mask_a = (offs_am[:, None] < M) & (offs_k[None, :] + k*BLOCK_SIZE_K < K)
                mask_b = (offs_k[:, None] + k*BLOCK_SIZE_K < K) & (offs_bn[None, :] < N)
                a = tl.load(A_ptrs, mask=mask_a, other=0.0)
                b = tl.load(B_ptrs, mask=mask_b, other=0.0)

                accum = tl.dot(a, b, accum)

                A_ptrs += BLOCK_SIZE_K*stride_ak
                B_ptrs += BLOCK_SIZE_K*stride_bk


            offs_c = offs_am[:, None]*stride_cm + offs_bn[None, :]*stride_cn
            mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

            tl.store(C_ptr+offs_c, accum, mask=mask_c)


@triton.autotune(
    configs=configs_any2D,
    key=['M', 'N'],
)
@triton.jit
def tadd(               
    A_ptr, B_ptr, C_ptr,
    M, N, 
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    ):                              
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        locations_a = (offs_m[:, None]*stride_am + offs_n[None, :]*stride_an)
        locations_b = (offs_m[:, None]*stride_bm + offs_n[None, :]*stride_bn)
        A_ptrs = locations_a + A_ptr
        B_ptrs = locations_b + B_ptr
        C_ptrs = locations_a + C_ptr

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        A = tl.load(A_ptrs, mask=mask, other=0.0)
        B = tl.load(B_ptrs, mask=mask, other=0.0)

        C = A + B

        tl.store(C_ptrs, C, mask=mask)


@triton.autotune(
    configs = configs_any2D,
    key = ['M', 'N'],
)
@triton.jit
def trelu(A_ptr, B_ptr, D,
        M, N,
        stride_am, stride_an,
        BLOCK_SIZE_M:tl.constexpr,
        BLOCK_SIZE_N:tl.constexpr):

        pid = tl.program_id(axis=0)

        num_block_rows = tl.cdiv(M, BLOCK_SIZE_M)
        num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

        pid_m = pid // num_block_cols
        pid_n = pid % num_block_cols

        offs_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        A_ptrs = A_ptr + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an

        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N) 

        A = tl.load(A_ptrs, mask=mask_a, other=0.0)

        accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accum2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        accum = tl.maximum(A, accum)
        accum2 = tl.where(accum>0.0, 1.0, 0.0)

        B_ptrs = B_ptr + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an
        D_ptrs = D + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an

        tl.store(B_ptrs, accum, mask=mask_a)
        tl.store(D_ptrs, accum2, mask=mask_a)

#TODO Fix exp range in tanh
@triton.autotune(
    configs=configs_any2D,
    key=['M', 'N'],
)
@triton.jit
def tanh(A_ptr, B_ptr,
        M, N,
        stride_am, stride_an,
        BLOCK_SIZE_M:tl.constexpr,
        BLOCK_SIZE_N:tl.constexpr):
        '''Unusable, there is uncentered exp, stabilize that first'''

        pid = tl.program_id(axis=0)

        total_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
        total_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

        pid_m = pid // total_blocks_n
        pid_n = pid % total_blocks_n

        offs_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        A_ptrs = A_ptr + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an
        B_ptrs = B_ptr + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an

        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        A = tl.load(A_ptrs, mask=mask_a, other=0.0)

        A2n = tl.exp(-2*A)
        
        n = 1 - A2n
        d = 1 + A2n

        final = tl.fdiv(n, d)

        tl.store(B_ptrs, final, mask=mask_a)
            


#IN SOFTMAX axis=1 (within a row) is the output for the current input, the row below has outputs for the next input
# so tehnically we are operating on rows.
# @triton.autotune(
#     configs=config_just_warps,
#     key=['M', 'N'],
# )
@triton.jit                     
def tsfx(
    A_ptr, Y_ptr, P_ptr, L_ptr,
    M, N,
    stride_am, stride_an,
    stride_ym, stride_yn,
    stride_pm, stride_pn,
    BLOCK_SIZE_N: tl.constexpr
):

    pid = tl.program_id(axis=0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    # Here BLOCK_SIZE_N needs to be pow of 2 (obviously) this is not a batching, each pid handles the entire row
    mask = col_offsets < N
    
    a_row_ptr = A_ptr + pid * stride_am + col_offsets * stride_an
    y_row_ptr = Y_ptr + pid * stride_ym + col_offsets * stride_yn
    p_row_ptr = P_ptr + pid * stride_pm + col_offsets * stride_pn
    
    l_ptr = L_ptr + pid 
    
    logits = tl.load(a_row_ptr, mask=mask, other=-float('inf'))
    y_true = tl.load(y_row_ptr, mask=mask, other=0.0)
    
    #NOTE AXIS=0 IN HERE, IS BECAUSE OUR A_PTRS LOAD A 1D TENSOR INTO THE MEMORY. IF tl.load had pointers that pointed to 2d shapes then we would have diff axes.
    m = tl.max(logits, axis=0)
    safe_logits = logits - m
    
    numerator = tl.exp(safe_logits)
    denominator = tl.sum(numerator, axis=0)
    
    p = numerator / denominator
    log_p = safe_logits - tl.log(denominator)   #log(exp(safe_logits)) = safe_logits
    
    # Cross Entropy Loss: Hadamard product and Sum
    loss  = -tl.sum(tl.where(mask, y_true * log_p, 0.0), axis=0)    #NOTE REAL LEARNING STUFF BECAUSE OF NAN (-inf*0)
    tl.store(p_row_ptr, p, mask=mask)
    tl.store(l_ptr, loss)

@triton.autotune(
    configs= configs_any2D, 
    key=["M", "N"],
)
@triton.jit
def thad(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_m, stride_n, 
    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n*BLOCK_SIZE_N +tl.arange(0, BLOCK_SIZE_N)

        off_block = offs_m[:, None]*stride_m + offs_n[None, :]*stride_n

        A_ptrs = A_ptr + off_block
        B_ptrs = B_ptr + off_block
        C_ptrs = C_ptr + off_block

        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        A = tl.load(A_ptrs, mask=mask_a, other=0.0)
        B = tl.load(B_ptrs, mask=mask_a, other=0.0)

        accum = A * B

        tl.store(C_ptrs, accum, mask=mask_a)
        


@triton.autotune(
    configs=matmul_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tmatxadd(
    A_ptr, B_ptr, C_ptr, D_ptr, E_ptr,              #E_ptr is the relu derivative, same shape as D_ptr
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K:tl.constexpr,
    GROUP_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        total_block_n = tl.cdiv(N, BLOCK_SIZE_N)
        total_block_m = tl.cdiv(M, BLOCK_SIZE_M)
        total_block_k = tl.cdiv(K, BLOCK_SIZE_K)

        total_pids_per_group = total_block_n * GROUP_SIZE
        group_id = pid // total_pids_per_group

        pid_idx_in_group = pid % total_pids_per_group

        group_start = group_id * GROUP_SIZE

        new_group_size = min(total_block_m - group_start, GROUP_SIZE)
        pid_m_in_group = pid_idx_in_group % new_group_size
        pid_n_in_group = pid_idx_in_group // new_group_size

        pid_m = pid_m_in_group + group_start 
        pid_n = pid_n_in_group


        offs_am = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        offs_k = tl.arange(0, BLOCK_SIZE_K)

        #Remember C here is given, its the Bias(most likely)

        offs_a = offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak
        offs_b = offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn

        offs_c = offs_am[:, None]*stride_cm + offs_bn[None, :]*stride_cn   #Remember that stride_cm ?= 0, in that case the offs_am have a row of zeros 
        offs_d = offs_am[:, None]*stride_dm + offs_bn[None, :]*stride_dn

        mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)    #This works, because C is === 2D same size as D. THe pointers are 2d and the rows are same, cuz of stride_cm ?=0

        mask_d = mask_c

        A_ptrs = A_ptr + offs_a
        B_ptrs = B_ptr + offs_b
        C_ptrs = C_ptr + offs_c

        accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accum2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(total_block_k):
            mask_a = (offs_am[:, None] < M) & (offs_k[None, :] + k*BLOCK_SIZE_K < K)
            mask_b = (offs_k[:, None] +k*BLOCK_SIZE_K < K) & (offs_bn[None, :] < N)

            a = tl.load(A_ptrs, mask=mask_a, other=0.0)
            b = tl.load(B_ptrs, mask=mask_b, other=0.0)

            accum = tl.dot(a, b, accum)

            A_ptrs += BLOCK_SIZE_K*stride_ak
            B_ptrs += BLOCK_SIZE_K*stride_bk

        c = tl.load(C_ptrs, mask=mask_c, other=0.0)

        accum += c

        accum = tl.maximum(accum, accum2)   #just a cheap shortcut to compare with zeros (accum2 should b zeros actually)
        accum2 = tl.where(accum>0.0, 1.0, 0.0)

        tl.store(E_ptr + offs_d, accum2, mask=mask_d)

        tl.store(D_ptr + offs_d, accum, mask=mask_d)


def TMUL(m1: torch.Tensor, m2: torch.Tensor, trans_a=False, trans_b=False)->torch.Tensor:

    m3 = None

    if not trans_a and not trans_b:
        M = m1.shape[0]
        K = m1.shape[1]
        assert K==m2.shape[0], "Matmul failed, since shapes dont match - 1"
        N = m2.shape[1]

        m3 = torch.empty((M,N), device=m1.device, dtype=torch.float32)
        grid = lambda meta : (triton.cdiv(M, meta["BLOCK_SIZE_M"])* triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        tmatmul[grid](m1, m2, m3, M, N, K, *m1.stride(), *m2.stride(), *m3.stride())

    elif trans_a and not trans_b:
        M = m1.shape[1]
        K = m1.shape[0]
        assert K==m2.shape[0], "Matmul failed, since shapes dont match - 2"
        N = m2.shape[1]

        m3 = torch.empty((M, N), device=m1.device, dtype=torch.float32)
        grid = lambda meta : (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        tmatmul[grid](m1, m2, m3, M, N, K, *reversed(m1.stride()), *m2.stride(), *m3.stride())

    elif not trans_a and trans_b:
        M = m1.shape[0]
        K = m1.shape[1]
        assert K==m2.shape[1], "Matmul failed , since shapes dont match - 3"
        N = m2.shape[0]

        m3 = torch.empty((M, N), device = m1.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        tmatmul[grid](m1, m2, m3, M, N, K, *m1.stride(), *reversed(m2.stride()), *m3.stride())

    elif trans_a and trans_b:
        M = m1.shape[1]
        K = m1.shape[0]
        assert K==m2.shape[1], "Matmul failed, since shapes dont match - 4"
        N = m2.shape[0]

        m3 = torch.empty((M, N), device = m1.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        tmatmul[grid](m1, m2, m3, M, N, K, *reversed(m1.stride()), *reversed(m2.stride()), *m3.stride())

    return m3


def TADD(m1: torch.Tensor, m2: torch.Tensor)->torch.Tensor:
    '''TODO should handle broadcasting along rows dimensions. assuming the rows are of different sizes'''
    m3 = torch.empty_like(m1, device=m1.device)
    M, N = m1.shape
    if m2.shape[0] == 1 and M != 1:
        stride_bm, stride_bn = 0, m2.stride(1)   # force row broadcast
    else:
        stride_bm, stride_bn = m2.stride()        # normal case, same shape

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) , triton.cdiv(N, meta["BLOCK_SIZE_N"]), )

    tadd[grid](m1, m2, m3, M, N, *m1.stride(), stride_bm, stride_bn)

    return m3



def TSFX(out: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Softmax + Xentropy, returns the final loss , and probability matrix after SOFTMAX'''
    P = torch.empty_like(out, device=out.device)
    L = torch.empty((P.shape[0], 1), device=out.device)    #L shape is 2D, so (Batchsize x 1)

    grid = (P.shape[0],)

    tsfx[grid](out, Y, P, L, *P.shape, *out.stride(), *Y.stride(), *P.stride(), triton.next_power_of_2(P.shape[1]))

    return L, P


def TRELU(m1: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor] :
    '''returns the relu of the matrix, as well as the gradient matrix.'''
    m2 = torch.empty_like(m1, device=m1.device)
    d = torch.empty_like(m1, device=m1.device)

    M, N= m1.shape

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    
    trelu[grid](m1, m2, d, *m1.shape, *m1.stride())

    return m2, d


def THAD(m1: torch.Tensor, m2:torch.Tensor)->torch.Tensor:
    '''returns the hadamard prod of the two matrices'''
    
    m3 = torch.empty_like(m1, device=m1.device)
    M, N = m1.shape

    grid = lambda meta : ((triton.cdiv(M, meta["BLOCK_SIZE_M"])), triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    thad[grid](m1, m2, m3, *m1.shape, *m1.stride())

    return m3


def TMATXADD(m1:torch.Tensor, m2:torch.Tensor, m3:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    M, K1 = m1.shape
    K, N = m2.shape

    assert K == K1, "TMATXADD has different sizes along the K dimension, fix the matmul shapes"
    assert N == m3.shape[1], "TMATXADD has different cols for bias addition to the matmul result"

    m4 = torch.empty((M, N), device=m1.device)
    m5 = torch.empty((M, N), device=m1.device)

    if m3.shape[0] == 1:
        stride_cm = 0
        stride_cn = m3.stride(1)
    else:
        stride_cm, stride_cn = m3.stride()


    grid = lambda meta : ((triton.cdiv(M, meta["BLOCK_SIZE_M"])*triton.cdiv(N, meta["BLOCK_SIZE_N"])), )

    tmatxadd[grid](m1, m2, m3, m4, m5, M, N, K, *m1.stride(), *m2.stride(), stride_cm, stride_cn, *m4.stride())
    return m4, m5