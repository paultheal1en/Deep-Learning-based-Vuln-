SCSIRequest *scsi_req_new(SCSIDevice *d, uint32_t tag, uint32_t lun,

                          uint8_t *buf, void *hba_private)

{

    SCSIRequest *req;

    req = d->info->alloc_req(d, tag, lun, hba_private);

    memcpy(req->cmd.buf, buf, 16);

    return req;

}
