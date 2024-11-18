static int scsi_write_data(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint32_t n;



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    n = r->iov.iov_len / 512;

    if (n) {

        qemu_iovec_init_external(&r->qiov, &r->iov, 1);

        r->req.aiocb = bdrv_aio_writev(s->bs, r->sector, &r->qiov, n,

                                   scsi_write_complete, r);

        if (r->req.aiocb == NULL) {

            scsi_write_complete(r, -EIO);

        }

    } else {

        /* Invoke completion routine to fetch data from host.  */

        scsi_write_complete(r, 0);

    }



    return 0;

}
