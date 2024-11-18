void scsi_device_purge_requests(SCSIDevice *sdev, SCSISense sense)

{

    SCSIRequest *req;



    while (!QTAILQ_EMPTY(&sdev->requests)) {

        req = QTAILQ_FIRST(&sdev->requests);

        scsi_req_cancel(req);

    }

    sdev->unit_attention = sense;

}
