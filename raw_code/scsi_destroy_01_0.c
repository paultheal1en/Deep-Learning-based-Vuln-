static void scsi_destroy(SCSIDevice *dev)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);



    scsi_disk_purge_requests(s);

    blockdev_mark_auto_del(s->qdev.conf.dinfo->bdrv);

}
