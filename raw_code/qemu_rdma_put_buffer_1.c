static int qemu_rdma_put_buffer(void *opaque, const uint8_t *buf,

                                int64_t pos, int size)

{

    QEMUFileRDMA *r = opaque;

    QEMUFile *f = r->file;

    RDMAContext *rdma = r->rdma;

    size_t remaining = size;

    uint8_t * data = (void *) buf;

    int ret;



    CHECK_ERROR_STATE();



    /*

     * Push out any writes that

     * we're queued up for VM's ram.

     */

    ret = qemu_rdma_write_flush(f, rdma);

    if (ret < 0) {

        rdma->error_state = ret;

        return ret;

    }



    while (remaining) {

        RDMAControlHeader head;



        r->len = MIN(remaining, RDMA_SEND_INCREMENT);

        remaining -= r->len;



        head.len = r->len;

        head.type = RDMA_CONTROL_QEMU_FILE;



        ret = qemu_rdma_exchange_send(rdma, &head, data, NULL, NULL, NULL);



        if (ret < 0) {

            rdma->error_state = ret;

            return ret;

        }



        data += r->len;

    }



    return size;

}
