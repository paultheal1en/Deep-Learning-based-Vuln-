static inline uint8_t *ram_chunk_end(const RDMALocalBlock *rdma_ram_block,

                                     uint64_t i)

{

    uint8_t *result = ram_chunk_start(rdma_ram_block, i) +

                                         (1UL << RDMA_REG_CHUNK_SHIFT);



    if (result > (rdma_ram_block->local_host_addr + rdma_ram_block->length)) {

        result = rdma_ram_block->local_host_addr + rdma_ram_block->length;

    }



    return result;

}
