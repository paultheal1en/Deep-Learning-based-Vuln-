static void scsi_aio_complete(void *opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    assert(r->req.aiocb != NULL);

    r->req.aiocb = NULL;

    block_acct_done(bdrv_get_stats(s->qdev.conf.bs), &r->acct);

    if (r->req.io_canceled) {

        scsi_req_cancel_complete(&r->req);

        goto done;

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    scsi_req_complete(&r->req, GOOD);



done:

    scsi_req_unref(&r->req);

}
