void scsi_req_free(SCSIRequest *req)

{

    scsi_req_dequeue(req);

    qemu_free(req);

}
