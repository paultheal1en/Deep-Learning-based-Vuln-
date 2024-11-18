iscsi_readcapacity16_cb(struct iscsi_context *iscsi, int status,

                        void *command_data, void *opaque)

{

    struct IscsiTask *itask = opaque;

    struct scsi_readcapacity16 *rc16;

    struct scsi_task *task = command_data;



    if (status != 0) {

        error_report("iSCSI: Failed to read capacity of iSCSI lun. %s",

                     iscsi_get_error(iscsi));

        itask->status   = 1;

        itask->complete = 1;

        scsi_free_scsi_task(task);

        return;

    }



    rc16 = scsi_datain_unmarshall(task);

    if (rc16 == NULL) {

        error_report("iSCSI: Failed to unmarshall readcapacity16 data.");

        itask->status   = 1;

        itask->complete = 1;

        scsi_free_scsi_task(task);

        return;

    }



    itask->iscsilun->block_size = rc16->block_length;

    itask->iscsilun->num_blocks = rc16->returned_lba + 1;

    itask->bs->total_sectors    = itask->iscsilun->num_blocks *

                               itask->iscsilun->block_size / BDRV_SECTOR_SIZE ;



    itask->status   = 0;

    itask->complete = 1;

    scsi_free_scsi_task(task);

}
