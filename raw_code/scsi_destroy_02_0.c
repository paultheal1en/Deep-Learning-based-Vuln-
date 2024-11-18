static void scsi_destroy(SCSIDevice *d)

{

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, d);

    SCSIGenericReq *r;



    while (!QTAILQ_EMPTY(&s->qdev.requests)) {

        r = DO_UPCAST(SCSIGenericReq, req, QTAILQ_FIRST(&s->qdev.requests));

        scsi_remove_request(r);

    }

    blockdev_mark_auto_del(s->qdev.conf.dinfo->bdrv);

}
