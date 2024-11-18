static void scsi_req_set_status(SCSIDiskReq *r, int status, int sense_code)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    r->req.status = status;

    scsi_disk_set_sense(s, sense_code);

}
