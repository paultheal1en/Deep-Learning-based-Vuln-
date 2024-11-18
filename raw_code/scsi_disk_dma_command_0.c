static int32_t scsi_disk_dma_command(SCSIRequest *req, uint8_t *buf)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    uint32_t len;

    uint8_t command;



    command = buf[0];



    if (s->tray_open || !bdrv_is_inserted(s->qdev.conf.bs)) {

        scsi_check_condition(r, SENSE_CODE(NO_MEDIUM));

        return 0;

    }



    len = scsi_data_cdb_length(r->req.cmd.buf);

    switch (command) {

    case READ_6:

    case READ_10:

    case READ_12:

    case READ_16:

        DPRINTF("Read (sector %" PRId64 ", count %u)\n", r->req.cmd.lba, len);

        if (r->req.cmd.buf[1] & 0xe0) {

            goto illegal_request;

        }

        if (!check_lba_range(s, r->req.cmd.lba, len)) {

            goto illegal_lba;

        }

        r->sector = r->req.cmd.lba * (s->qdev.blocksize / 512);

        r->sector_count = len * (s->qdev.blocksize / 512);

        break;

    case WRITE_6:

    case WRITE_10:

    case WRITE_12:

    case WRITE_16:

    case WRITE_VERIFY_10:

    case WRITE_VERIFY_12:

    case WRITE_VERIFY_16:

        if (bdrv_is_read_only(s->qdev.conf.bs)) {

            scsi_check_condition(r, SENSE_CODE(WRITE_PROTECTED));

            return 0;

        }

        DPRINTF("Write %s(sector %" PRId64 ", count %u)\n",

                (command & 0xe) == 0xe ? "And Verify " : "",

                r->req.cmd.lba, len);

        if (r->req.cmd.buf[1] & 0xe0) {

            goto illegal_request;

        }

        if (!check_lba_range(s, r->req.cmd.lba, len)) {

            goto illegal_lba;

        }

        r->sector = r->req.cmd.lba * (s->qdev.blocksize / 512);

        r->sector_count = len * (s->qdev.blocksize / 512);

        break;

    default:

        abort();

    illegal_request:

        scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

        return 0;

    illegal_lba:

        scsi_check_condition(r, SENSE_CODE(LBA_OUT_OF_RANGE));

        return 0;

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
