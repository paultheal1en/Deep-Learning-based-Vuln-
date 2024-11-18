int scsi_req_get_sense(SCSIRequest *req, uint8_t *buf, int len)

{

    assert(len >= 14);

    if (!req->sense_len) {

        return 0;

    }

    return scsi_build_sense(req->sense, req->sense_len, buf, len, true);

}
