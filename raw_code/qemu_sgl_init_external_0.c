static void qemu_sgl_init_external(VirtIOSCSIReq *req, struct iovec *sg,

                                   hwaddr *addr, int num)

{

    QEMUSGList *qsgl = &req->qsgl;



    qemu_sglist_init(qsgl, DEVICE(req->dev), num, &address_space_memory);

    while (num--) {

        qemu_sglist_add(qsgl, *(addr++), (sg++)->iov_len);

    }

}
