static int32_t scsi_send_command(SCSIRequest *req, uint8_t *buf)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    int32_t len;

    uint8_t command;

    uint8_t *outbuf;

    int rc;



    command = buf[0];

    outbuf = (uint8_t *)r->iov.iov_base;

    DPRINTF("Command: lun=%d tag=0x%x data=0x%02x", req->lun, req->tag, buf[0]);



    if (scsi_req_parse(&r->req, buf) != 0) {

        BADF("Unsupported command length, command %x\n", command);

        scsi_command_complete(r, CHECK_CONDITION, SENSE_CODE(INVALID_OPCODE));

        return 0;

    }

#ifdef DEBUG_SCSI

    {

        int i;

        for (i = 1; i < r->req.cmd.len; i++) {

            printf(" 0x%02x", buf[i]);

        }

        printf("\n");

    }

#endif



    if (req->lun) {

        /* Only LUN 0 supported.  */

        DPRINTF("Unimplemented LUN %d\n", req->lun);

        if (command != REQUEST_SENSE && command != INQUIRY) {

            scsi_command_complete(r, CHECK_CONDITION,

                                  SENSE_CODE(LUN_NOT_SUPPORTED));

            return 0;

        }

    }

    switch (command) {

    case TEST_UNIT_READY:

    case REQUEST_SENSE:

    case INQUIRY:

    case MODE_SENSE:

    case MODE_SENSE_10:

    case RESERVE:

    case RESERVE_10:

    case RELEASE:

    case RELEASE_10:

    case START_STOP:

    case ALLOW_MEDIUM_REMOVAL:

    case READ_CAPACITY:

    case SYNCHRONIZE_CACHE:

    case READ_TOC:

    case GET_CONFIGURATION:

    case SERVICE_ACTION_IN:

    case REPORT_LUNS:

    case VERIFY:

        rc = scsi_disk_emulate_command(r, outbuf);

        if (rc < 0) {

            return 0;

        }



        r->iov.iov_len = rc;

        break;

    case READ_6:

    case READ_10:

    case READ_12:

    case READ_16:

        len = r->req.cmd.xfer / s->qdev.blocksize;

        DPRINTF("Read (sector %" PRId64 ", count %d)\n", r->req.cmd.lba, len);

        if (r->req.cmd.lba > s->max_lba)

            goto illegal_lba;

        r->sector = r->req.cmd.lba * s->cluster_size;

        r->sector_count = len * s->cluster_size;

        break;

    case WRITE_6:

    case WRITE_10:

    case WRITE_12:

    case WRITE_16:

    case WRITE_VERIFY:

    case WRITE_VERIFY_12:

    case WRITE_VERIFY_16:

        len = r->req.cmd.xfer / s->qdev.blocksize;

        DPRINTF("Write %s(sector %" PRId64 ", count %d)\n",

                (command & 0xe) == 0xe ? "And Verify " : "",

                r->req.cmd.lba, len);

        if (r->req.cmd.lba > s->max_lba)

            goto illegal_lba;

        r->sector = r->req.cmd.lba * s->cluster_size;

        r->sector_count = len * s->cluster_size;

        break;

    case MODE_SELECT:

        DPRINTF("Mode Select(6) (len %lu)\n", (long)r->req.cmd.xfer);

        /* We don't support mode parameter changes.

           Allow the mode parameter header + block descriptors only. */

        if (r->req.cmd.xfer > 12) {

            goto fail;

        }

        break;

    case MODE_SELECT_10:

        DPRINTF("Mode Select(10) (len %lu)\n", (long)r->req.cmd.xfer);

        /* We don't support mode parameter changes.

           Allow the mode parameter header + block descriptors only. */

        if (r->req.cmd.xfer > 16) {

            goto fail;

        }

        break;

    case SEEK_6:

    case SEEK_10:

        DPRINTF("Seek(%d) (sector %" PRId64 ")\n", command == SEEK_6 ? 6 : 10,

                r->req.cmd.lba);

        if (r->req.cmd.lba > s->max_lba) {

            goto illegal_lba;

        }

        break;

    case WRITE_SAME_16:

        len = r->req.cmd.xfer / s->qdev.blocksize;



        DPRINTF("WRITE SAME(16) (sector %" PRId64 ", count %d)\n",

                r->req.cmd.lba, len);



        if (r->req.cmd.lba > s->max_lba) {

            goto illegal_lba;

        }



        /*

         * We only support WRITE SAME with the unmap bit set for now.

         */

        if (!(buf[1] & 0x8)) {

            goto fail;

        }



        rc = bdrv_discard(s->bs, r->req.cmd.lba * s->cluster_size,

                          len * s->cluster_size);

        if (rc < 0) {

            /* XXX: better error code ?*/

            goto fail;

        }



        break;

    default:

        DPRINTF("Unknown SCSI command (%2.2x)\n", buf[0]);

        scsi_command_complete(r, CHECK_CONDITION, SENSE_CODE(INVALID_OPCODE));

        return 0;

    fail:

        scsi_command_complete(r, CHECK_CONDITION, SENSE_CODE(INVALID_FIELD));

        return 0;

    illegal_lba:

        scsi_command_complete(r, CHECK_CONDITION, SENSE_CODE(LBA_OUT_OF_RANGE));

        return 0;

    }

    if (r->sector_count == 0 && r->iov.iov_len == 0) {

        scsi_command_complete(r, GOOD, SENSE_CODE(NO_SENSE));

    }

    len = r->sector_count * 512 + r->iov.iov_len;

    if (r->req.cmd.mode == SCSI_XFER_TO_DEV) {

        return -len;

    } else {

        if (!r->sector_count)

            r->sector_count = -1;

        return len;

    }

}
