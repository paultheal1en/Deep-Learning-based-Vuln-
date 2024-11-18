static int scsi_generic_initfn(SCSIDevice *dev)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, dev);

    int sg_version;

    struct sg_scsi_id scsiid;



    if (!s->qdev.conf.dinfo || !s->qdev.conf.dinfo->bdrv) {

        error_report("scsi-generic: drive property not set");

        return -1;

    }

    s->bs = s->qdev.conf.dinfo->bdrv;



    /* check we are really using a /dev/sg* file */

    if (!bdrv_is_sg(s->bs)) {

        error_report("scsi-generic: not /dev/sg*");

        return -1;

    }



    /* check we are using a driver managing SG_IO (version 3 and after */

    if (bdrv_ioctl(s->bs, SG_GET_VERSION_NUM, &sg_version) < 0 ||

        sg_version < 30000) {

        error_report("scsi-generic: scsi generic interface too old");

        return -1;

    }



    /* get LUN of the /dev/sg? */

    if (bdrv_ioctl(s->bs, SG_GET_SCSI_ID, &scsiid)) {

        error_report("scsi-generic: SG_GET_SCSI_ID ioctl failed");

        return -1;

    }



    /* define device state */

    s->lun = scsiid.lun;

    DPRINTF("LUN %d\n", s->lun);

    s->qdev.type = scsiid.scsi_type;

    DPRINTF("device type %d\n", s->qdev.type);

    if (s->qdev.type == TYPE_TAPE) {

        s->qdev.blocksize = get_stream_blocksize(s->bs);

        if (s->qdev.blocksize == -1)

            s->qdev.blocksize = 0;

    } else {

        s->qdev.blocksize = get_blocksize(s->bs);

        /* removable media returns 0 if not present */

        if (s->qdev.blocksize <= 0) {

            if (s->qdev.type == TYPE_ROM || s->qdev.type  == TYPE_WORM)

                s->qdev.blocksize = 2048;

            else

                s->qdev.blocksize = 512;

        }

    }

    DPRINTF("block size %d\n", s->qdev.blocksize);

    s->driver_status = 0;

    memset(s->sensebuf, 0, sizeof(s->sensebuf));

    return 0;

}
