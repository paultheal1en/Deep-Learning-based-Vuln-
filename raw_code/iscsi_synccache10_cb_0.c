iscsi_synccache10_cb(struct iscsi_context *iscsi, int status,

                     void *command_data, void *opaque)

{

    IscsiAIOCB *acb = opaque;



    if (acb->canceled != 0) {

        return;

    }



    acb->status = 0;

    if (status != 0) {

        if (status == SCSI_STATUS_CHECK_CONDITION

            && acb->task->sense.key == SCSI_SENSE_UNIT_ATTENTION

            && acb->retries-- > 0) {

            if (acb->task != NULL) {

                scsi_free_scsi_task(acb->task);

                acb->task = NULL;

            }

            if (iscsi_aio_flush_acb(acb) == 0) {

                iscsi_set_events(acb->iscsilun);

                return;

            }

        }

        error_report("Failed to sync10 data on iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        acb->status = -EIO;

    }



    iscsi_schedule_bh(acb);

}
