static void scsi_flush_complete(void * opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    if (r->req.aiocb != NULL) {

        r->req.aiocb = NULL;

        bdrv_acct_done(s->qdev.conf.bs, &r->acct);

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    scsi_req_complete(&r->req, GOOD);



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
