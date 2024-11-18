static void qemu_rdma_signal_unregister(RDMAContext *rdma, uint64_t index,

                                        uint64_t chunk, uint64_t wr_id)

{

    if (rdma->unregistrations[rdma->unregister_next] != 0) {

        fprintf(stderr, "rdma migration: queue is full!\n");

    } else {

        RDMALocalBlock *block = &(rdma->local_ram_blocks.block[index]);



        if (!test_and_set_bit(chunk, block->unregister_bitmap)) {

            DDPRINTF("Appending unregister chunk %" PRIu64

                    " at position %d\n", chunk, rdma->unregister_next);



            rdma->unregistrations[rdma->unregister_next++] =

                    qemu_rdma_make_wrid(wr_id, index, chunk);



            if (rdma->unregister_next == RDMA_SIGNALED_SEND_MAX) {

                rdma->unregister_next = 0;

            }

        } else {

            DDPRINTF("Unregister chunk %" PRIu64 " already in queue.\n",

                    chunk);

        }

    }

}
