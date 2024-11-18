static inline uint64_t ram_chunk_index(const uint8_t *start,

                                       const uint8_t *host)

{

    return ((uintptr_t) host - (uintptr_t) start) >> RDMA_REG_CHUNK_SHIFT;

}
