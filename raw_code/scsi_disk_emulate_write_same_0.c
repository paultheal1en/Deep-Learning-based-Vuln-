static void scsi_disk_emulate_write_same(SCSIDiskReq *r, uint8_t *inbuf)

{

    SCSIRequest *req = &r->req;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    uint32_t nb_sectors = scsi_data_cdb_length(r->req.cmd.buf);

    WriteSameCBData *data;

    uint8_t *buf;

    int i;



    /* Fail if PBDATA=1 or LBDATA=1 or ANCHOR=1.  */

    if (nb_sectors == 0 || (req->cmd.buf[1] & 0x16)) {

        scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

        return;

    }



    if (bdrv_is_read_only(s->qdev.conf.bs)) {

        scsi_check_condition(r, SENSE_CODE(WRITE_PROTECTED));

        return;

    }

    if (!check_lba_range(s, r->req.cmd.lba, nb_sectors)) {

        scsi_check_condition(r, SENSE_CODE(LBA_OUT_OF_RANGE));

        return;

    }



    if (buffer_is_zero(inbuf, s->qdev.blocksize)) {

        int flags = (req->cmd.buf[1] & 0x8) ? BDRV_REQ_MAY_UNMAP : 0;



        /* The request is used as the AIO opaque value, so add a ref.  */

        scsi_req_ref(&r->req);

        block_acct_start(bdrv_get_stats(s->qdev.conf.bs), &r->acct,

                         nb_sectors * s->qdev.blocksize,

                        BLOCK_ACCT_WRITE);

        r->req.aiocb = bdrv_aio_write_zeroes(s->qdev.conf.bs,

                                             r->req.cmd.lba * (s->qdev.blocksize / 512),

                                             nb_sectors * (s->qdev.blocksize / 512),

                                             flags, scsi_aio_complete, r);

        return;

    }



    data = g_new0(WriteSameCBData, 1);

    data->r = r;

    data->sector = r->req.cmd.lba * (s->qdev.blocksize / 512);

    data->nb_sectors = nb_sectors * (s->qdev.blocksize / 512);

    data->iov.iov_len = MIN(data->nb_sectors * 512, SCSI_WRITE_SAME_MAX);

    data->iov.iov_base = buf = qemu_blockalign(s->qdev.conf.bs, data->iov.iov_len);

    qemu_iovec_init_external(&data->qiov, &data->iov, 1);



    for (i = 0; i < data->iov.iov_len; i += s->qdev.blocksize) {

        memcpy(&buf[i], inbuf, s->qdev.blocksize);

    }



    scsi_req_ref(&r->req);

    block_acct_start(bdrv_get_stats(s->qdev.conf.bs), &r->acct,

                     data->iov.iov_len, BLOCK_ACCT_WRITE);

    r->req.aiocb = bdrv_aio_writev(s->qdev.conf.bs, data->sector,

                                   &data->qiov, data->iov.iov_len / 512,

                                   scsi_write_same_complete, data);

}
