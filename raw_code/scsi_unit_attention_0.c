static int32_t scsi_unit_attention(SCSIRequest *req, uint8_t *buf)

{

    if (req->dev && req->dev->unit_attention.key == UNIT_ATTENTION) {

        scsi_req_build_sense(req, req->dev->unit_attention);

    } else if (req->bus->unit_attention.key == UNIT_ATTENTION) {

        scsi_req_build_sense(req, req->bus->unit_attention);

    }

    scsi_req_complete(req, CHECK_CONDITION);

    return 0;

}
