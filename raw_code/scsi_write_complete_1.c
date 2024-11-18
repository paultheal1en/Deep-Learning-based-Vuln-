static void scsi_write_complete(void * opaque, int ret)

{

    SCSIDiskReq *r = (SCSIDiskReq *)opaque;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



    if (r->req.aiocb != NULL) {

        r->req.aiocb = NULL;

        bdrv_acct_done(s->qdev.conf.bs, &r->acct);

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    n = r->qiov.size / 512;

    r->sector += n;

    r->sector_count -= n;

    if (r->sector_count == 0) {

        scsi_req_complete(&r->req, GOOD);

    } else {

        scsi_init_iovec(r, SCSI_DMA_BUF_SIZE);

        DPRINTF("Write complete tag=0x%x more=%d\n", r->req.tag, r->qiov.size);

        scsi_req_data(&r->req, r->qiov.size);

    }



done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
