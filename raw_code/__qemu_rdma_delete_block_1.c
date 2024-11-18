static int __qemu_rdma_delete_block(RDMAContext *rdma, ram_addr_t block_offset)

{

    RDMALocalBlocks *local = &rdma->local_ram_blocks;

    RDMALocalBlock *block = g_hash_table_lookup(rdma->blockmap,

        (void *) block_offset);

    RDMALocalBlock *old = local->block;

    int x;



    assert(block);



    if (block->pmr) {

        int j;



        for (j = 0; j < block->nb_chunks; j++) {

            if (!block->pmr[j]) {

                continue;

            }

            ibv_dereg_mr(block->pmr[j]);

            rdma->total_registrations--;

        }

        g_free(block->pmr);

        block->pmr = NULL;

    }



    if (block->mr) {

        ibv_dereg_mr(block->mr);

        rdma->total_registrations--;

        block->mr = NULL;

    }



    g_free(block->transit_bitmap);

    block->transit_bitmap = NULL;



    g_free(block->unregister_bitmap);

    block->unregister_bitmap = NULL;



    g_free(block->remote_keys);

    block->remote_keys = NULL;



    for (x = 0; x < local->nb_blocks; x++) {

        g_hash_table_remove(rdma->blockmap, (void *)old[x].offset);

    }



    if (local->nb_blocks > 1) {



        local->block = g_malloc0(sizeof(RDMALocalBlock) *

                                    (local->nb_blocks - 1));



        if (block->index) {

            memcpy(local->block, old, sizeof(RDMALocalBlock) * block->index);

        }



        if (block->index < (local->nb_blocks - 1)) {

            memcpy(local->block + block->index, old + (block->index + 1),

                sizeof(RDMALocalBlock) *

                    (local->nb_blocks - (block->index + 1)));

        }

    } else {

        assert(block == local->block);

        local->block = NULL;

    }



    DDPRINTF("Deleted Block: %d, addr: %" PRIu64 ", offset: %" PRIu64

           " length: %" PRIu64 " end: %" PRIu64 " bits %" PRIu64 " chunks %d\n",

            local->nb_blocks, (uint64_t) block->local_host_addr, block->offset,

            block->length, (uint64_t) (block->local_host_addr + block->length),

                BITS_TO_LONGS(block->nb_chunks) *

                    sizeof(unsigned long) * 8, block->nb_chunks);



    g_free(old);



    local->nb_blocks--;



    if (local->nb_blocks) {

        for (x = 0; x < local->nb_blocks; x++) {

            g_hash_table_insert(rdma->blockmap, (void *)local->block[x].offset,

                                                &local->block[x]);

        }

    }



    return 0;

}
