static void iscsi_readcapacity_sync(IscsiLun *iscsilun, Error **errp)

{

    struct scsi_task *task = NULL;

    struct scsi_readcapacity10 *rc10 = NULL;

    struct scsi_readcapacity16 *rc16 = NULL;

    int retries = ISCSI_CMD_RETRIES; 



    do {

        if (task != NULL) {

            scsi_free_scsi_task(task);

            task = NULL;

        }



        switch (iscsilun->type) {

        case TYPE_DISK:

            task = iscsi_readcapacity16_sync(iscsilun->iscsi, iscsilun->lun);

            if (task != NULL && task->status == SCSI_STATUS_GOOD) {

                rc16 = scsi_datain_unmarshall(task);

                if (rc16 == NULL) {

                    error_setg(errp, "iSCSI: Failed to unmarshall readcapacity16 data.");

                } else {

                    iscsilun->block_size = rc16->block_length;

                    iscsilun->num_blocks = rc16->returned_lba + 1;

                    iscsilun->lbpme = !!rc16->lbpme;

                    iscsilun->lbprz = !!rc16->lbprz;

                    iscsilun->use_16_for_rw = (rc16->returned_lba > 0xffffffff);

                }

            }

            break;

        case TYPE_ROM:

            task = iscsi_readcapacity10_sync(iscsilun->iscsi, iscsilun->lun, 0, 0);

            if (task != NULL && task->status == SCSI_STATUS_GOOD) {

                rc10 = scsi_datain_unmarshall(task);

                if (rc10 == NULL) {

                    error_setg(errp, "iSCSI: Failed to unmarshall readcapacity10 data.");

                } else {

                    iscsilun->block_size = rc10->block_size;

                    if (rc10->lba == 0) {

                        /* blank disk loaded */

                        iscsilun->num_blocks = 0;

                    } else {

                        iscsilun->num_blocks = rc10->lba + 1;

                    }

                }

            }

            break;

        default:

            return;

        }

    } while (task != NULL && task->status == SCSI_STATUS_CHECK_CONDITION

             && task->sense.key == SCSI_SENSE_UNIT_ATTENTION

             && retries-- > 0);



    if (task == NULL || task->status != SCSI_STATUS_GOOD) {

        error_setg(errp, "iSCSI: failed to send readcapacity10 command.");





    }

    if (task) {

        scsi_free_scsi_task(task);

    }

}