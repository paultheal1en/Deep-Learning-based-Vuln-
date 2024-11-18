static void scsi_cancel_io(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);



    DPRINTF("Cancel tag=0x%x\n", req->tag);

    if (r->req.aiocb) {

        bdrv_aio_cancel(r->req.aiocb);



        /* This reference was left in by scsi_*_data.  We take ownership of

         * it the moment scsi_req_cancel is called, independent of whether

         * bdrv_aio_cancel completes the request or not.  */

        scsi_req_unref(&r->req);

    }

    r->req.aiocb = NULL;

}
