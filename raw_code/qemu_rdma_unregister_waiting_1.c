static int qemu_rdma_unregister_waiting(RDMAContext *rdma)

{

    while (rdma->unregistrations[rdma->unregister_current]) {

        int ret;

        uint64_t wr_id = rdma->unregistrations[rdma->unregister_current];

        uint64_t chunk =

            (wr_id & RDMA_WRID_CHUNK_MASK) >> RDMA_WRID_CHUNK_SHIFT;

        uint64_t index =

            (wr_id & RDMA_WRID_BLOCK_MASK) >> RDMA_WRID_BLOCK_SHIFT;

        RDMALocalBlock *block =

            &(rdma->local_ram_blocks.block[index]);

        RDMARegister reg = { .current_index = index };

        RDMAControlHeader resp = { .type = RDMA_CONTROL_UNREGISTER_FINISHED,

                                 };

        RDMAControlHeader head = { .len = sizeof(RDMARegister),

                                   .type = RDMA_CONTROL_UNREGISTER_REQUEST,

                                   .repeat = 1,

                                 };



        DDPRINTF("Processing unregister for chunk: %" PRIu64

                 " at position %d\n", chunk, rdma->unregister_current);



        rdma->unregistrations[rdma->unregister_current] = 0;

        rdma->unregister_current++;



        if (rdma->unregister_current == RDMA_SIGNALED_SEND_MAX) {

            rdma->unregister_current = 0;

        }





        /*

         * Unregistration is speculative (because migration is single-threaded

         * and we cannot break the protocol's inifinband message ordering).

         * Thus, if the memory is currently being used for transmission,

         * then abort the attempt to unregister and try again

         * later the next time a completion is received for this memory.

         */

        clear_bit(chunk, block->unregister_bitmap);



        if (test_bit(chunk, block->transit_bitmap)) {

            DDPRINTF("Cannot unregister inflight chunk: %" PRIu64 "\n", chunk);

            continue;

        }



        DDPRINTF("Sending unregister for chunk: %" PRIu64 "\n", chunk);



        ret = ibv_dereg_mr(block->pmr[chunk]);

        block->pmr[chunk] = NULL;

        block->remote_keys[chunk] = 0;



        if (ret != 0) {

            perror("unregistration chunk failed");

            return -ret;

        }

        rdma->total_registrations--;



        reg.key.chunk = chunk;

        register_to_network(&reg);

        ret = qemu_rdma_exchange_send(rdma, &head, (uint8_t *) &reg,

                                &resp, NULL, NULL);

        if (ret < 0) {

            return ret;

        }



        DDPRINTF("Unregister for chunk: %" PRIu64 " complete.\n", chunk);

    }



    return 0;

}
