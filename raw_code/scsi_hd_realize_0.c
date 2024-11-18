static void scsi_hd_realize(SCSIDevice *dev, Error **errp)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    blkconf_blocksizes(&s->qdev.conf);

    s->qdev.blocksize = s->qdev.conf.logical_block_size;

    s->qdev.type = TYPE_DISK;

    if (!s->product) {

        s->product = g_strdup("QEMU HARDDISK");

    }

    scsi_realize(&s->qdev, errp);

}
