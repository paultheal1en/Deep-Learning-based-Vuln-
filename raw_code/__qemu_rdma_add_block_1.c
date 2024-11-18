static int __qemu_rdma_add_block(RDMAContext *rdma, void *host_addr,

                         ram_addr_t block_offset, uint64_t length)

{

    RDMALocalBlocks *local = &rdma->local_ram_blocks;

    RDMALocalBlock *block = g_hash_table_lookup(rdma->blockmap,

        (void *) block_offset);

    RDMALocalBlock *old = local->block;



    assert(block == NULL);



    local->block = g_malloc0(sizeof(RDMALocalBlock) * (local->nb_blocks + 1));



    if (local->nb_blocks) {

        int x;



        for (x = 0; x < local->nb_blocks; x++) {

            g_hash_table_remove(rdma->blockmap, (void *)old[x].offset);

            g_hash_table_insert(rdma->blockmap, (void *)old[x].offset,

                                                &local->block[x]);

        }

        memcpy(local->block, old, sizeof(RDMALocalBlock) * local->nb_blocks);

        g_free(old);

    }



    block = &local->block[local->nb_blocks];



    block->local_host_addr = host_addr;

    block->offset = block_offset;

    block->length = length;

    block->index = local->nb_blocks;

    block->nb_chunks = ram_chunk_index(host_addr, host_addr + length) + 1UL;

    block->transit_bitmap = bitmap_new(block->nb_chunks);

    bitmap_clear(block->transit_bitmap, 0, block->nb_chunks);

    block->unregister_bitmap = bitmap_new(block->nb_chunks);

    bitmap_clear(block->unregister_bitmap, 0, block->nb_chunks);

    block->remote_keys = g_malloc0(block->nb_chunks * sizeof(uint32_t));



    block->is_ram_block = local->init ? false : true;



    g_hash_table_insert(rdma->blockmap, (void *) block_offset, block);



    DDPRINTF("Added Block: %d, addr: %" PRIu64 ", offset: %" PRIu64

           " length: %" PRIu64 " end: %" PRIu64 " bits %" PRIu64 " chunks %d\n",

            local->nb_blocks, (uint64_t) block->local_host_addr, block->offset,

            block->length, (uint64_t) (block->local_host_addr + block->length),

                BITS_TO_LONGS(block->nb_chunks) *

                    sizeof(unsigned long) * 8, block->nb_chunks);



    local->nb_blocks++;



    return 0;

}
