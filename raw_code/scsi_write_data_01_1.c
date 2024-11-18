static int scsi_write_data(SCSIDevice *d, uint32_t tag)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, d);

    SCSIGenericReq *r;

    int ret;



    DPRINTF("scsi_write_data 0x%x\n", tag);

    r = scsi_find_request(s, tag);

    if (!r) {

        BADF("Bad write tag 0x%x\n", tag);

        /* ??? This is the wrong error.  */

        scsi_command_complete(r, -EINVAL);

        return 0;

    }



    if (r->len == 0) {

        r->len = r->buflen;

        scsi_req_data(&r->req, r->len);

        return 0;

    }



    ret = execute_command(s->bs, r, SG_DXFER_TO_DEV, scsi_write_complete);

    if (ret == -1) {

        scsi_command_complete(r, -EINVAL);

        return 1;

    }



    return 0;

}
