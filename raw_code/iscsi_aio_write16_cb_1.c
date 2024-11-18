iscsi_aio_write16_cb(struct iscsi_context *iscsi, int status,

                     void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    trace_iscsi_aio_write16_cb(iscsi, status, acb, acb->canceled);



    g_free(acb->buf);



    if (acb->canceled != 0) {

        qemu_aio_release(acb);

        scsi_free_scsi_task(acb->task);

        acb->task = NULL;

        return;

    }



    acb->status = 0;

    if (status < 0) {

        error_report("Failed to write16 data to iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    iscsi_schedule_bh(acb);

    scsi_free_scsi_task(acb->task);

    acb->task = NULL;

}
