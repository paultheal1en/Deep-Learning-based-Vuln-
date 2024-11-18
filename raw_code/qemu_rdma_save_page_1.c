static size_t qemu_rdma_save_page(QEMUFile *f, void *opaque,

                                  ram_addr_t block_offset, ram_addr_t offset,

                                  size_t size, int *bytes_sent)

{

    QEMUFileRDMA *rfile = opaque;

    RDMAContext *rdma = rfile->rdma;

    int ret;



    CHECK_ERROR_STATE();



    qemu_fflush(f);



    if (size > 0) {

        /*

         * Add this page to the current 'chunk'. If the chunk

         * is full, or the page doen't belong to the current chunk,

         * an actual RDMA write will occur and a new chunk will be formed.

         */

        ret = qemu_rdma_write(f, rdma, block_offset, offset, size);

        if (ret < 0) {

            fprintf(stderr, "rdma migration: write error! %d\n", ret);

            goto err;

        }



        /*

         * We always return 1 bytes because the RDMA

         * protocol is completely asynchronous. We do not yet know

         * whether an  identified chunk is zero or not because we're

         * waiting for other pages to potentially be merged with

         * the current chunk. So, we have to call qemu_update_position()

         * later on when the actual write occurs.

         */

        if (bytes_sent) {

            *bytes_sent = 1;

        }

    } else {

        uint64_t index, chunk;



        /* TODO: Change QEMUFileOps prototype to be signed: size_t => long

        if (size < 0) {

            ret = qemu_rdma_drain_cq(f, rdma);

            if (ret < 0) {

                fprintf(stderr, "rdma: failed to synchronously drain"

                                " completion queue before unregistration.\n");

                goto err;

            }

        }

        */



        ret = qemu_rdma_search_ram_block(rdma, block_offset,

                                         offset, size, &index, &chunk);



        if (ret) {

            fprintf(stderr, "ram block search failed\n");

            goto err;

        }



        qemu_rdma_signal_unregister(rdma, index, chunk, 0);



        /*

         * TODO: Synchronous, guaranteed unregistration (should not occur during

         * fast-path). Otherwise, unregisters will process on the next call to

         * qemu_rdma_drain_cq()

        if (size < 0) {

            qemu_rdma_unregister_waiting(rdma);

        }

        */

    }



    /*

     * Drain the Completion Queue if possible, but do not block,

     * just poll.

     *

     * If nothing to poll, the end of the iteration will do this

     * again to make sure we don't overflow the request queue.

     */

    while (1) {

        uint64_t wr_id, wr_id_in;

        int ret = qemu_rdma_poll(rdma, &wr_id_in, NULL);

        if (ret < 0) {

            fprintf(stderr, "rdma migration: polling error! %d\n", ret);

            goto err;

        }



        wr_id = wr_id_in & RDMA_WRID_TYPE_MASK;



        if (wr_id == RDMA_WRID_NONE) {

            break;

        }

    }



    return RAM_SAVE_CONTROL_DELAYED;

err:

    rdma->error_state = ret;

    return ret;

}
