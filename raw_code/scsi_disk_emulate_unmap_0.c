static void scsi_disk_emulate_unmap(SCSIDiskReq *r, uint8_t *inbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint8_t *p = inbuf;

    int len = r->req.cmd.xfer;

    UnmapCBData *data;



    /* Reject ANCHOR=1.  */

    if (r->req.cmd.buf[1] & 0x1) {

        goto invalid_field;

    }



    if (len < 8) {

        goto invalid_param_len;

    }

    if (len < lduw_be_p(&p[0]) + 2) {

        goto invalid_param_len;

    }

    if (len < lduw_be_p(&p[2]) + 8) {

        goto invalid_param_len;

    }

    if (lduw_be_p(&p[2]) & 15) {

        goto invalid_param_len;

    }



    if (bdrv_is_read_only(s->qdev.conf.bs)) {

        scsi_check_condition(r, SENSE_CODE(WRITE_PROTECTED));

        return;

    }



    data = g_new0(UnmapCBData, 1);

    data->r = r;

    data->inbuf = &p[8];

    data->count = lduw_be_p(&p[2]) >> 4;



    /* The matching unref is in scsi_unmap_complete, before data is freed.  */

    scsi_req_ref(&r->req);

    scsi_unmap_complete(data, 0);

    return;



invalid_param_len:

    scsi_check_condition(r, SENSE_CODE(INVALID_PARAM_LEN));

    return;



invalid_field:

    scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

}
