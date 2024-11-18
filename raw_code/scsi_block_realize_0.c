static void scsi_block_realize(SCSIDevice *dev, Error **errp)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    int sg_version;

    int rc;



    if (!s->qdev.conf.bs) {

        error_setg(errp, "drive property not set");

        return;

    }



    /* check we are using a driver managing SG_IO (version 3 and after) */

    rc = bdrv_ioctl(s->qdev.conf.bs, SG_GET_VERSION_NUM, &sg_version);

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



    /* get device type from INQUIRY data */

    rc = get_device_type(s);

    if (rc < 0) {

        error_setg(errp, "INQUIRY failed");

        return;

    }



    /* Make a guess for the block size, we'll fix it when the guest sends.

     * READ CAPACITY.  If they don't, they likely would assume these sizes

     * anyway. (TODO: check in /sys).

     */

    if (s->qdev.type == TYPE_ROM || s->qdev.type == TYPE_WORM) {

        s->qdev.blocksize = 2048;

    } else {

        s->qdev.blocksize = 512;

    }



    /* Makes the scsi-block device not removable by using HMP and QMP eject

     * command.

     */

    s->features |= (1 << SCSI_DISK_F_NO_REMOVABLE_DEVOPS);



    scsi_realize(&s->qdev, errp);

}
