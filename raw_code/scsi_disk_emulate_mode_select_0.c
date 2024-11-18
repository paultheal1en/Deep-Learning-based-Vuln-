static void scsi_disk_emulate_mode_select(SCSIDiskReq *r, uint8_t *inbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint8_t *p = inbuf;

    int cmd = r->req.cmd.buf[0];

    int len = r->req.cmd.xfer;

    int hdr_len = (cmd == MODE_SELECT ? 4 : 8);

    int bd_len;

    int pass;



    /* We only support PF=1, SP=0.  */

    if ((r->req.cmd.buf[1] & 0x11) != 0x10) {

        goto invalid_field;

    }



    if (len < hdr_len) {

        goto invalid_param_len;

    }



    bd_len = (cmd == MODE_SELECT ? p[3] : lduw_be_p(&p[6]));

    len -= hdr_len;

    p += hdr_len;

    if (len < bd_len) {

        goto invalid_param_len;

    }

    if (bd_len != 0 && bd_len != 8) {

        goto invalid_param;

    }



    len -= bd_len;

    p += bd_len;



    /* Ensure no change is made if there is an error!  */

    for (pass = 0; pass < 2; pass++) {

        if (mode_select_pages(r, p, len, pass == 1) < 0) {

            assert(pass == 0);

            return;

        }

    }

    if (!bdrv_enable_write_cache(s->qdev.conf.bs)) {

        /* The request is used as the AIO opaque value, so add a ref.  */

        scsi_req_ref(&r->req);

        block_acct_start(bdrv_get_stats(s->qdev.conf.bs), &r->acct, 0,

                         BLOCK_ACCT_FLUSH);

        r->req.aiocb = bdrv_aio_flush(s->qdev.conf.bs, scsi_aio_complete, r);

        return;

    }



    scsi_req_complete(&r->req, GOOD);

    return;



invalid_param:

    scsi_check_condition(r, SENSE_CODE(INVALID_PARAM));

    return;



invalid_param_len:

    scsi_check_condition(r, SENSE_CODE(INVALID_PARAM_LEN));

    return;



invalid_field:

    scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

}
