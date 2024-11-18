static int scsi_disk_emulate_command(SCSIDiskReq *r)

{

    SCSIRequest *req = &r->req;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    uint64_t nb_sectors;

    uint8_t *outbuf;

    int buflen = 0;



    if (!r->iov.iov_base) {

        /*

         * FIXME: we shouldn't return anything bigger than 4k, but the code

         * requires the buffer to be as big as req->cmd.xfer in several

         * places.  So, do not allow CDBs with a very large ALLOCATION

         * LENGTH.  The real fix would be to modify scsi_read_data and

         * dma_buf_read, so that they return data beyond the buflen

         * as all zeros.

         */

        if (req->cmd.xfer > 65536) {

            goto illegal_request;

        }

        r->buflen = MAX(4096, req->cmd.xfer);

        r->iov.iov_base = qemu_blockalign(s->qdev.conf.bs, r->buflen);

    }



    outbuf = r->iov.iov_base;

    switch (req->cmd.buf[0]) {

    case TEST_UNIT_READY:

        assert(!s->tray_open && bdrv_is_inserted(s->qdev.conf.bs));

        break;

    case INQUIRY:

        buflen = scsi_disk_emulate_inquiry(req, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case MODE_SENSE:

    case MODE_SENSE_10:

        buflen = scsi_disk_emulate_mode_sense(r, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case READ_TOC:

        buflen = scsi_disk_emulate_read_toc(req, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case RESERVE:

        if (req->cmd.buf[1] & 1) {

            goto illegal_request;

        }

        break;

    case RESERVE_10:

        if (req->cmd.buf[1] & 3) {

            goto illegal_request;

        }

        break;

    case RELEASE:

        if (req->cmd.buf[1] & 1) {

            goto illegal_request;

        }

        break;

    case RELEASE_10:

        if (req->cmd.buf[1] & 3) {

            goto illegal_request;

        }

        break;

    case START_STOP:

        if (scsi_disk_emulate_start_stop(r) < 0) {

            return -1;

        }

        break;

    case ALLOW_MEDIUM_REMOVAL:

        s->tray_locked = req->cmd.buf[4] & 1;

        bdrv_lock_medium(s->qdev.conf.bs, req->cmd.buf[4] & 1);

        break;

    case READ_CAPACITY_10:

        /* The normal LEN field for this command is zero.  */

        memset(outbuf, 0, 8);

        bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);

        if (!nb_sectors) {

            scsi_check_condition(r, SENSE_CODE(LUN_NOT_READY));

            return -1;

        }

        if ((req->cmd.buf[8] & 1) == 0 && req->cmd.lba) {

            goto illegal_request;

        }

        nb_sectors /= s->qdev.blocksize / 512;

        /* Returned value is the address of the last sector.  */

        nb_sectors--;

        /* Remember the new size for read/write sanity checking. */

        s->qdev.max_lba = nb_sectors;

        /* Clip to 2TB, instead of returning capacity modulo 2TB. */

        if (nb_sectors > UINT32_MAX) {

            nb_sectors = UINT32_MAX;

        }

        outbuf[0] = (nb_sectors >> 24) & 0xff;

        outbuf[1] = (nb_sectors >> 16) & 0xff;

        outbuf[2] = (nb_sectors >> 8) & 0xff;

        outbuf[3] = nb_sectors & 0xff;

        outbuf[4] = 0;

        outbuf[5] = 0;

        outbuf[6] = s->qdev.blocksize >> 8;

        outbuf[7] = 0;

        buflen = 8;

        break;

    case REQUEST_SENSE:

        /* Just return "NO SENSE".  */

        buflen = scsi_build_sense(NULL, 0, outbuf, r->buflen,

                                  (req->cmd.buf[1] & 1) == 0);

        break;

    case MECHANISM_STATUS:

        buflen = scsi_emulate_mechanism_status(s, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case GET_CONFIGURATION:

        buflen = scsi_get_configuration(s, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case GET_EVENT_STATUS_NOTIFICATION:

        buflen = scsi_get_event_status_notification(s, r, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case READ_DVD_STRUCTURE:

        buflen = scsi_read_dvd_structure(s, r, outbuf);

        if (buflen < 0) {

            goto illegal_request;

        }

        break;

    case SERVICE_ACTION_IN_16:

        /* Service Action In subcommands. */

        if ((req->cmd.buf[1] & 31) == SAI_READ_CAPACITY_16) {

            DPRINTF("SAI READ CAPACITY(16)\n");

            memset(outbuf, 0, req->cmd.xfer);

            bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);

            if (!nb_sectors) {

                scsi_check_condition(r, SENSE_CODE(LUN_NOT_READY));

                return -1;

            }

            if ((req->cmd.buf[14] & 1) == 0 && req->cmd.lba) {

                goto illegal_request;

            }

            nb_sectors /= s->qdev.blocksize / 512;

            /* Returned value is the address of the last sector.  */

            nb_sectors--;

            /* Remember the new size for read/write sanity checking. */

            s->qdev.max_lba = nb_sectors;

            outbuf[0] = (nb_sectors >> 56) & 0xff;

            outbuf[1] = (nb_sectors >> 48) & 0xff;

            outbuf[2] = (nb_sectors >> 40) & 0xff;

            outbuf[3] = (nb_sectors >> 32) & 0xff;

            outbuf[4] = (nb_sectors >> 24) & 0xff;

            outbuf[5] = (nb_sectors >> 16) & 0xff;

            outbuf[6] = (nb_sectors >> 8) & 0xff;

            outbuf[7] = nb_sectors & 0xff;

            outbuf[8] = 0;

            outbuf[9] = 0;

            outbuf[10] = s->qdev.blocksize >> 8;

            outbuf[11] = 0;

            outbuf[12] = 0;

            outbuf[13] = get_physical_block_exp(&s->qdev.conf);



            /* set TPE bit if the format supports discard */

            if (s->qdev.conf.discard_granularity) {

                outbuf[14] = 0x80;

            }



            /* Protection, exponent and lowest lba field left blank. */

            buflen = req->cmd.xfer;

            break;

        }

        DPRINTF("Unsupported Service Action In\n");

        goto illegal_request;

    case VERIFY_10:

        break;

    default:

        scsi_check_condition(r, SENSE_CODE(INVALID_OPCODE));

        return -1;

    }

    buflen = MIN(buflen, req->cmd.xfer);

    return buflen;



illegal_request:

    if (r->req.status == -1) {

        scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

    }

    return -1;

}
