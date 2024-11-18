static void rbd_aio_bh_cb(void *opaque)

{

    RBDAIOCB *acb = opaque;



    if (acb->cmd == RBD_AIO_READ) {

        qemu_iovec_from_buf(acb->qiov, 0, acb->bounce, acb->qiov->size);

    }

    qemu_vfree(acb->bounce);

    acb->common.cb(acb->common.opaque, (acb->ret > 0 ? 0 : acb->ret));

    qemu_bh_delete(acb->bh);

    acb->bh = NULL;



    qemu_aio_release(acb);

}
