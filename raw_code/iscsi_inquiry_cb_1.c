iscsi_inquiry_cb(struct iscsi_context *iscsi, int status, void *command_data,

                 void *opaque)

{

    struct IscsiTask *itask = opaque;

    struct scsi_task *task = command_data;

    struct scsi_inquiry_standard *inq;



    if (status != 0) {

        itask->status   = 1;

        itask->complete = 1;

        scsi_free_scsi_task(task);

        return;

    }



    inq = scsi_datain_unmarshall(task);

    if (inq == NULL) {

        error_report("iSCSI: Failed to unmarshall inquiry data.");

        itask->status   = 1;

        itask->complete = 1;

        scsi_free_scsi_task(task);

        return;

    }



    itask->iscsilun->type = inq->periperal_device_type;



    scsi_free_scsi_task(task);



    switch (itask->iscsilun->type) {

    case TYPE_DISK:

        task = iscsi_readcapacity16_task(iscsi, itask->iscsilun->lun,

                                   iscsi_readcapacity16_cb, opaque);

        if (task == NULL) {

            error_report("iSCSI: failed to send readcapacity16 command.");

            itask->status   = 1;

            itask->complete = 1;

            return;

        }

        break;

    case TYPE_ROM:

        task = iscsi_readcapacity10_task(iscsi, itask->iscsilun->lun,

                                   0, 0,

                                   iscsi_readcapacity10_cb, opaque);

        if (task == NULL) {

            error_report("iSCSI: failed to send readcapacity16 command.");

            itask->status   = 1;

            itask->complete = 1;

            return;

        }

        break;

    default:

        itask->status   = 0;

        itask->complete = 1;

    }

}
