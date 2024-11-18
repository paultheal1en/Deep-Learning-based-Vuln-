pvscsi_command_complete(SCSIRequest *req, uint32_t status, size_t resid)

{

    PVSCSIRequest *pvscsi_req = req->hba_private;

    PVSCSIState *s = pvscsi_req->dev;



    if (!pvscsi_req) {

        trace_pvscsi_command_complete_not_found(req->tag);

        return;

    }



    if (resid) {

        /* Short transfer.  */

        trace_pvscsi_command_complete_data_run();

        pvscsi_req->cmp.hostStatus = BTSTAT_DATARUN;

    }



    pvscsi_req->cmp.scsiStatus = status;

    if (pvscsi_req->cmp.scsiStatus == CHECK_CONDITION) {

        uint8_t sense[SCSI_SENSE_BUF_SIZE];

        int sense_len =

            scsi_req_get_sense(pvscsi_req->sreq, sense, sizeof(sense));



        trace_pvscsi_command_complete_sense_len(sense_len);

        pvscsi_write_sense(pvscsi_req, sense, sense_len);

    }

    qemu_sglist_destroy(&pvscsi_req->sgl);

    pvscsi_complete_request(s, pvscsi_req);

}
