static int scsi_hd_initfn(SCSIDevice *dev)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    s->qdev.blocksize = s->qdev.conf.logical_block_size;

    s->qdev.type = TYPE_DISK;

    if (!s->product) {

        s->product = g_strdup("QEMU HARDDISK");

    }

    return scsi_initfn(&s->qdev);

}
