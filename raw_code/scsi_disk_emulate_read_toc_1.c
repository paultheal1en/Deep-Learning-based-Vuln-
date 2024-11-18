static int scsi_disk_emulate_read_toc(SCSIRequest *req, uint8_t *outbuf)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    int start_track, format, msf, toclen;

    uint64_t nb_sectors;



    msf = req->cmd.buf[1] & 2;

    format = req->cmd.buf[2] & 0xf;

    start_track = req->cmd.buf[6];

    bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);

    DPRINTF("Read TOC (track %d format %d msf %d)\n", start_track, format, msf >> 1);

    nb_sectors /= s->qdev.blocksize / 512;

    switch (format) {

    case 0:

        toclen = cdrom_read_toc(nb_sectors, outbuf, msf, start_track);

        break;

    case 1:

        /* multi session : only a single session defined */

        toclen = 12;

        memset(outbuf, 0, 12);

        outbuf[1] = 0x0a;

        outbuf[2] = 0x01;

        outbuf[3] = 0x01;

        break;

    case 2:

        toclen = cdrom_read_toc_raw(nb_sectors, outbuf, msf, start_track);

        break;

    default:

        return -1;

    }

    if (toclen > req->cmd.xfer) {

        toclen = req->cmd.xfer;

    }

    return toclen;

}
