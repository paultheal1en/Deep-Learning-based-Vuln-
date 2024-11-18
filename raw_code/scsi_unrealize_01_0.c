static void scsi_unrealize(SCSIDevice *dev, Error **errp)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);



    scsi_device_purge_requests(&s->qdev, SENSE_CODE(NO_SENSE));

    blockdev_mark_auto_del(s->qdev.conf.blk);

}
