static SCSIGenericReq *scsi_new_request(SCSIDevice *d, uint32_t tag, uint32_t lun)

{

    SCSIRequest *req;



    req = scsi_req_alloc(sizeof(SCSIGenericReq), d, tag, lun);

    return DO_UPCAST(SCSIGenericReq, req, req);

}
