static uint8_t *scsi_get_buf(SCSIDevice *d, uint32_t tag)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, d);

    SCSIDiskReq *r;



    r = scsi_find_request(s, tag);

    if (!r) {

        BADF("Bad buffer tag 0x%x\n", tag);

        return NULL;

    }

    return (uint8_t *)r->iov.iov_base;

}
