static void scsi_write_data(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    SCSIDiskClass *sdc = (SCSIDiskClass *) object_get_class(OBJECT(s));



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    /* The request is used as the AIO opaque value, so add a ref.  */

    scsi_req_ref(&r->req);

    if (r->req.cmd.mode != SCSI_XFER_TO_DEV) {

        DPRINTF("Data transfer direction invalid\n");

        scsi_write_complete_noio(r, -EINVAL);

        return;

    }



    if (!r->req.sg && !r->qiov.size) {

        /* Called for the first time.  Ask the driver to send us more data.  */

        r->started = true;

        scsi_write_complete_noio(r, 0);

        return;

    }

    if (s->tray_open) {

        scsi_write_complete_noio(r, -ENOMEDIUM);

        return;

    }



    if (r->req.cmd.buf[0] == VERIFY_10 || r->req.cmd.buf[0] == VERIFY_12 ||

        r->req.cmd.buf[0] == VERIFY_16) {

        if (r->req.sg) {

            scsi_dma_complete_noio(r, 0);

        } else {

            scsi_write_complete_noio(r, 0);

        }

        return;

    }



    if (r->req.sg) {

        dma_acct_start(s->qdev.conf.blk, &r->acct, r->req.sg, BLOCK_ACCT_WRITE);

        r->req.resid -= r->req.sg->size;

        r->req.aiocb = dma_blk_io(blk_get_aio_context(s->qdev.conf.blk),

                                  r->req.sg, r->sector << BDRV_SECTOR_BITS,

                                  sdc->dma_writev, r, scsi_dma_complete, r,

                                  DMA_DIRECTION_TO_DEVICE);

    } else {

        block_acct_start(blk_get_stats(s->qdev.conf.blk), &r->acct,

                         r->qiov.size, BLOCK_ACCT_WRITE);

        r->req.aiocb = sdc->dma_writev(r->sector << BDRV_SECTOR_BITS, &r->qiov,

                                       scsi_write_complete, r, r);

    }

}
