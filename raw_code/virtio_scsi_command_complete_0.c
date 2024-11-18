static void virtio_scsi_command_complete(SCSIRequest *r, uint32_t status,

                                         size_t resid)

{

    VirtIOSCSIReq *req = r->hba_private;

    uint8_t sense[SCSI_SENSE_BUF_SIZE];

    uint32_t sense_len;



    if (r->io_canceled) {

        return;

    }



    req->resp.cmd->response = VIRTIO_SCSI_S_OK;

    req->resp.cmd->status = status;

    if (req->resp.cmd->status == GOOD) {

        req->resp.cmd->resid = tswap32(resid);

    } else {

        req->resp.cmd->resid = 0;

        sense_len = scsi_req_get_sense(r, sense, sizeof(sense));

        sense_len = MIN(sense_len, req->resp_size - sizeof(req->resp.cmd));

        memcpy(req->resp.cmd->sense, sense, sense_len);

        req->resp.cmd->sense_len = tswap32(sense_len);

    }

    virtio_scsi_complete_cmd_req(req);

}
