/*
 * cxl_tp.h — 张量并行共享内存基础设施
 *
 * 每个 TP rank 是一个独立进程，通过 mmap 共享内存做 all-reduce。
 * 调用 ggml_tp_shm_init() 后，ggml 的 TP collective ops 自动可用。
 */

#ifndef CXL_TP_H
#define CXL_TP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 初始化 TP 共享内存上下文
 *
 * @param rank       当前 TP rank (0 ~ size-1)
 * @param size       TP 总 rank 数 (4)
 * @param shm_base   共享内存基址（所有 rank mmap 同一块物理页）
 * @param shm_size   共享内存大小（字节）
 *
 * 调用时机：ggml 图计算前，每个 TP 进程调用一次。
 */
void ggml_tp_shm_init(int rank, int size, void * shm_base, size_t shm_size);

#ifdef __cplusplus
}
#endif

#endif /* CXL_TP_H */
