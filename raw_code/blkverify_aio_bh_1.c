static void blkverify_aio_bh(void *opaque)

{

    BlkverifyAIOCB *acb = opaque;



    if (acb->buf) {

        qemu_iovec_destroy(&acb->raw_qiov);

        qemu_vfree(acb->buf);

    }

    acb->common.cb(acb->common.opaque, acb->ret);

    qemu_aio_unref(acb);

}
