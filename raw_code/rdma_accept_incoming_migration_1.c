static void rdma_accept_incoming_migration(void *opaque)

{

    RDMAContext *rdma = opaque;

    int ret;

    QEMUFile *f;

    Error *local_err = NULL, **errp = &local_err;



    DPRINTF("Accepting rdma connection...\n");

    ret = qemu_rdma_accept(rdma);



    if (ret) {

        ERROR(errp, "RDMA Migration initialization failed!");

        return;

    }



    DPRINTF("Accepted migration\n");



    f = qemu_fopen_rdma(rdma, "rb");

    if (f == NULL) {

        ERROR(errp, "could not qemu_fopen_rdma!");

        qemu_rdma_cleanup(rdma);

        return;

    }



    rdma->migration_started_on_destination = 1;

    process_incoming_migration(f);

}
