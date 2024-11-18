static int qemu_rdma_write_flush(QEMUFile *f, RDMAContext *rdma)

{

    int ret;



    if (!rdma->current_length) {

        return 0;

    }



    ret = qemu_rdma_write_one(f, rdma,

            rdma->current_index, rdma->current_addr, rdma->current_length);



    if (ret < 0) {

        return ret;

    }



    if (ret == 0) {

        rdma->nb_sent++;

        DDDPRINTF("sent total: %d\n", rdma->nb_sent);

    }



    rdma->current_length = 0;

    rdma->current_addr = 0;



    return 0;

}
