static void scsi_generic_realize(SCSIDevice *s, Error **errp)

{

    int rc;

    int sg_version;

    struct sg_scsi_id scsiid;



    if (!s->conf.bs) {

        error_setg(errp, "drive property not set");

        return;

    }



    if (bdrv_get_on_error(s->conf.bs, 0) != BLOCKDEV_ON_ERROR_ENOSPC) {

        error_setg(errp, "Device doesn't support drive option werror");

        return;

    }

    if (bdrv_get_on_error(s->conf.bs, 1) != BLOCKDEV_ON_ERROR_REPORT) {

        error_setg(errp, "Device doesn't support drive option rerror");

        return;

    }



    /* check we are using a driver managing SG_IO (version 3 and after */

    rc = bdrv_ioctl(s->conf.bs, SG_GET_VERSION_NUM, &sg_version);

    if (rc < 0) {

        error_setg(errp, "cannot get SG_IO version number: %s.  "

                         "Is this a SCSI device?",

                         strerror(-rc));

        return;

    }

    if (sg_version < 30000) {

        error_setg(errp, "scsi generic interface too old");

        return;

    }



    /* get LUN of the /dev/sg? */

    if (bdrv_ioctl(s->conf.bs, SG_GET_SCSI_ID, &scsiid)) {

        error_setg(errp, "SG_GET_SCSI_ID ioctl failed");

        return;

    }



    /* define device state */

    s->type = scsiid.scsi_type;

    DPRINTF("device type %d\n", s->type);



    switch (s->type) {

    case TYPE_TAPE:

        s->blocksize = get_stream_blocksize(s->conf.bs);

        if (s->blocksize == -1) {

            s->blocksize = 0;

        }

        break;



        /* Make a guess for block devices, we'll fix it when the guest sends.

         * READ CAPACITY.  If they don't, they likely would assume these sizes

         * anyway. (TODO: they could also send MODE SENSE).

         */

    case TYPE_ROM:

    case TYPE_WORM:

        s->blocksize = 2048;

        break;

    default:

        s->blocksize = 512;

        break;

    }



    DPRINTF("block size %d\n", s->blocksize);

}
