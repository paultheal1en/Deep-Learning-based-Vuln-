static int qemu_rdma_search_ram_block(RDMAContext *rdma,

                                      uint64_t block_offset,

                                      uint64_t offset,

                                      uint64_t length,

                                      uint64_t *block_index,

                                      uint64_t *chunk_index)

{

    uint64_t current_addr = block_offset + offset;

    RDMALocalBlock *block = g_hash_table_lookup(rdma->blockmap,

                                                (void *) block_offset);

    assert(block);

    assert(current_addr >= block->offset);

    assert((current_addr + length) <= (block->offset + block->length));



    *block_index = block->index;

    *chunk_index = ram_chunk_index(block->local_host_addr,

                block->local_host_addr + (current_addr - block->offset));



    return 0;

}
