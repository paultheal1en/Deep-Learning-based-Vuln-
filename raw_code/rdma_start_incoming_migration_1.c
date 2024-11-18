void rdma_start_incoming_migration(const char *host_port, Error **errp)

{

    int ret;

    RDMAContext *rdma;

    Error *local_err = NULL;



    DPRINTF("Starting RDMA-based incoming migration\n");

    rdma = qemu_rdma_data_init(host_port, &local_err);



    if (rdma == NULL) {

        goto err;

    }



    ret = qemu_rdma_dest_init(rdma, &local_err);



    if (ret) {

        goto err;

    }



    DPRINTF("qemu_rdma_dest_init success\n");



    ret = rdma_listen(rdma->listen_id, 5);



    if (ret) {

        ERROR(errp, "listening on socket!");

        goto err;

    }



    DPRINTF("rdma_listen success\n");



    qemu_set_fd_handler2(rdma->channel->fd, NULL,

                         rdma_accept_incoming_migration, NULL,

                            (void *)(intptr_t) rdma);

    return;

err:

    error_propagate(errp, local_err);

    g_free(rdma);

}
