static void scsi_realize(SCSIDevice *dev, Error **errp)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    Error *err = NULL;



    if (!s->qdev.conf.bs) {

        error_setg(errp, "drive property not set");

        return;

    }



    if (!(s->features & (1 << SCSI_DISK_F_REMOVABLE)) &&

        !bdrv_is_inserted(s->qdev.conf.bs)) {

        error_setg(errp, "Device needs media, but drive is empty");

        return;

    }



    blkconf_serial(&s->qdev.conf, &s->serial);

    if (dev->type == TYPE_DISK) {

        blkconf_geometry(&dev->conf, NULL, 65535, 255, 255, &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

    }



    if (s->qdev.conf.discard_granularity == -1) {

        s->qdev.conf.discard_granularity =

            MAX(s->qdev.conf.logical_block_size, DEFAULT_DISCARD_GRANULARITY);

    }



    if (!s->version) {

        s->version = g_strdup(qemu_get_version());

    }

    if (!s->vendor) {

        s->vendor = g_strdup("QEMU");

    }



    if (bdrv_is_sg(s->qdev.conf.bs)) {

        error_setg(errp, "unwanted /dev/sg*");

        return;

    }



    if ((s->features & (1 << SCSI_DISK_F_REMOVABLE)) &&

            !(s->features & (1 << SCSI_DISK_F_NO_REMOVABLE_DEVOPS))) {

        bdrv_set_dev_ops(s->qdev.conf.bs, &scsi_disk_removable_block_ops, s);

    } else {

        bdrv_set_dev_ops(s->qdev.conf.bs, &scsi_disk_block_ops, s);

    }

    bdrv_set_guest_block_size(s->qdev.conf.bs, s->qdev.blocksize);



    bdrv_iostatus_enable(s->qdev.conf.bs);

}
