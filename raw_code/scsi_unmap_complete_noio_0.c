static void scsi_unmap_complete_noio(UnmapCBData *data, int ret)

{

    SCSIDiskReq *r = data->r;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint64_t sector_num;

    uint32_t nb_sectors;



    assert(r->req.aiocb == NULL);



    if (r->req.io_canceled) {

        scsi_req_cancel_complete(&r->req);

        goto done;

    }



    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret, false)) {

            goto done;

        }

    }



    if (data->count > 0) {

        sector_num = ldq_be_p(&data->inbuf[0]);

        nb_sectors = ldl_be_p(&data->inbuf[8]) & 0xffffffffULL;

        if (!check_lba_range(s, sector_num, nb_sectors)) {

            scsi_check_condition(r, SENSE_CODE(LBA_OUT_OF_RANGE));

            goto done;

        }



        r->req.aiocb = blk_aio_discard(s->qdev.conf.blk,

                                       sector_num * (s->qdev.blocksize / 512),

                                       nb_sectors * (s->qdev.blocksize / 512),

                                       scsi_unmap_complete, data);

        data->count--;

        data->inbuf += 16;

        return;

    }



    scsi_req_complete(&r->req, GOOD);



done:

    scsi_req_unref(&r->req);

    g_free(data);

}
