void scsi_req_abort(SCSIRequest *req, int status)

{

    if (!req->enqueued) {

        return;

    }

    scsi_req_ref(req);

    scsi_req_dequeue(req);

    req->io_canceled = true;

    if (req->ops->cancel_io) {

        req->ops->cancel_io(req);

    }

    scsi_req_complete(req, status);

    scsi_req_unref(req);

}
