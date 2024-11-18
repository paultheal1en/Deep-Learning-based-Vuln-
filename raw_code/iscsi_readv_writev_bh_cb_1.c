iscsi_readv_writev_bh_cb(void *p)

{

    IscsiAIOCB *acb = p;



    qemu_bh_delete(acb->bh);



    if (!acb->canceled) {

        acb->common.cb(acb->common.opaque, acb->status);

    }



    qemu_aio_release(acb);



    if (acb->canceled) {

        return;

    }



    scsi_free_scsi_task(acb->task);

    acb->task = NULL;

}
