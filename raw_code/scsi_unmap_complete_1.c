static void scsi_unmap_complete(void *opaque, int ret)

{

    UnmapCBData *data = opaque;

    SCSIDiskReq *r = data->r;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);

    uint64_t sector_num;

    uint32_t nb_sectors;



    r->req.aiocb = NULL;

    if (ret < 0) {

        if (scsi_handle_rw_error(r, -ret)) {

            goto done;

        }

    }



    if (data->count > 0 && !r->req.io_canceled) {

        sector_num = ldq_be_p(&data->inbuf[0]);

        nb_sectors = ldl_be_p(&data->inbuf[8]) & 0xffffffffULL;

        if (sector_num > sector_num + nb_sectors ||

            sector_num + nb_sectors - 1 > s->qdev.max_lba) {

            scsi_check_condition(r, SENSE_CODE(LBA_OUT_OF_RANGE));

            goto done;

        }



        r->req.aiocb = bdrv_aio_discard(s->qdev.conf.bs,

                                        sector_num * (s->qdev.blocksize / 512),

                                        nb_sectors * (s->qdev.blocksize / 512),

                                        scsi_unmap_complete, data);

        data->count--;

        data->inbuf += 16;

        return;

    }



done:

    if (data->count == 0) {

        scsi_req_complete(&r->req, GOOD);

    }

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

    g_free(data);

}
