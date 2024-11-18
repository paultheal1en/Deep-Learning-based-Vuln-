static void scsi_generic_purge_requests(SCSIGenericState *s)

{

    SCSIGenericReq *r;



    while (!QTAILQ_EMPTY(&s->qdev.requests)) {

        r = DO_UPCAST(SCSIGenericReq, req, QTAILQ_FIRST(&s->qdev.requests));

        if (r->req.aiocb) {

            bdrv_aio_cancel(r->req.aiocb);

        }

        scsi_remove_request(r);

    }

}
