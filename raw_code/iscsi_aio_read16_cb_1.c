iscsi_aio_read16_cb(struct iscsi_context *iscsi, int status,

                    void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    trace_iscsi_aio_read16_cb(iscsi, status, acb, acb->canceled);



    if (acb->canceled) {

        qemu_aio_release(acb);

        return;

    }



    acb->status = 0;

    if (status != 0) {

        error_report("Failed to read16 data from iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    iscsi_schedule_bh(iscsi_readv_writev_bh_cb, acb);

}
