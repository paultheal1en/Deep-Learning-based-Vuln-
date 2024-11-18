static int qemu_rdma_source_init(RDMAContext *rdma, Error **errp, bool pin_all)

{

    int ret, idx;

    Error *local_err = NULL, **temp = &local_err;



    /*

     * Will be validated against destination's actual capabilities

     * after the connect() completes.

     */

    rdma->pin_all = pin_all;



    ret = qemu_rdma_resolve_host(rdma, temp);

    if (ret) {

        goto err_rdma_source_init;

    }



    ret = qemu_rdma_alloc_pd_cq(rdma);

    if (ret) {

        ERROR(temp, "rdma migration: error allocating pd and cq! Your mlock()"

                    " limits may be too low. Please check $ ulimit -a # and "

                    "search for 'ulimit -l' in the output");

        goto err_rdma_source_init;

    }



    ret = qemu_rdma_alloc_qp(rdma);

    if (ret) {

        ERROR(temp, "rdma migration: error allocating qp!");

        goto err_rdma_source_init;

    }



    ret = qemu_rdma_init_ram_blocks(rdma);

    if (ret) {

        ERROR(temp, "rdma migration: error initializing ram blocks!");

        goto err_rdma_source_init;

    }



    for (idx = 0; idx < RDMA_WRID_MAX; idx++) {

        ret = qemu_rdma_reg_control(rdma, idx);

        if (ret) {

            ERROR(temp, "rdma migration: error registering %d control!",

                                                            idx);

            goto err_rdma_source_init;

        }

    }



    return 0;



err_rdma_source_init:

    error_propagate(errp, local_err);

    qemu_rdma_cleanup(rdma);

    return -1;

}
