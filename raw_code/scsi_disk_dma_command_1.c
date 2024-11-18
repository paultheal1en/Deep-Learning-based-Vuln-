static int32_t scsi_disk_dma_command(SCSIRequest *req, uint8_t *buf)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    int32_t len;

    uint8_t command;



    command = buf[0];



    if (s->tray_open || !bdrv_is_inserted(s->qdev.conf.bs)) {

        scsi_check_condition(r, SENSE_CODE(NO_MEDIUM));


    }



    switch (command) {

    case READ_6:

    case READ_10:

    case READ_12:

    case READ_16:

        len = r->req.cmd.xfer / s->qdev.blocksize;

        DPRINTF("Read (sector %" PRId64 ", count %d)\n", r->req.cmd.lba, len);

        if (r->req.cmd.buf[1] & 0xe0) {

            goto illegal_request;

        }

        if (r->req.cmd.lba > s->qdev.max_lba) {

            goto illegal_lba;

        }

        r->sector = r->req.cmd.lba * (s->qdev.blocksize / 512);

        r->sector_count = len * (s->qdev.blocksize / 512);

        break;

    case VERIFY_10:

    case VERIFY_12:

    case VERIFY_16:

    case WRITE_6:

    case WRITE_10:

    case WRITE_12:

    case WRITE_16:

    case WRITE_VERIFY_10:

    case WRITE_VERIFY_12:

    case WRITE_VERIFY_16:

        len = r->req.cmd.xfer / s->qdev.blocksize;

        DPRINTF("Write %s(sector %" PRId64 ", count %d)\n",

                (command & 0xe) == 0xe ? "And Verify " : "",

                r->req.cmd.lba, len);

        if (r->req.cmd.buf[1] & 0xe0) {

            goto illegal_request;

        }

        if (r->req.cmd.lba > s->qdev.max_lba) {

            goto illegal_lba;

        }

        r->sector = r->req.cmd.lba * (s->qdev.blocksize / 512);

        r->sector_count = len * (s->qdev.blocksize / 512);

        break;

    default:

        abort();




    illegal_lba:

        scsi_check_condition(r, SENSE_CODE(LBA_OUT_OF_RANGE));


    }

    if (r->sector_count == 0) {

        scsi_req_complete(&r->req, GOOD);

    }

    assert(r->iov.iov_len == 0);

    if (r->req.cmd.mode == SCSI_XFER_TO_DEV) {

        return -r->sector_count * 512;

    } else {

        return r->sector_count * 512;

    }

}