iscsi_aio_flush(BlockDriverState *bs,

                BlockDriverCompletionFunc *cb, void *opaque)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;

    IscsiAIOCB *acb;



    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);



    acb->iscsilun = iscsilun;

    acb->canceled   = 0;

    acb->bh         = NULL;

    acb->status     = -EINPROGRESS;

    acb->buf        = NULL;



    acb->task = iscsi_synchronizecache10_task(iscsi, iscsilun->lun,

                                         0, 0, 0, 0,

                                         iscsi_synccache10_cb,

                                         acb);

    if (acb->task == NULL) {

        error_report("iSCSI: Failed to send synchronizecache10 command. %s",

                     iscsi_get_error(iscsi));

        qemu_aio_release(acb);

        return NULL;

    }



    iscsi_set_events(iscsilun);



    return &acb->common;

}
