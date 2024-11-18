static void scsi_command_complete(SCSIDiskReq *r, int status, int sense)

{

    DPRINTF("Command complete tag=0x%x status=%d sense=%d\n",

            r->req.tag, status, sense);

    scsi_req_set_status(r, status, sense);

    scsi_req_complete(&r->req);

    scsi_remove_request(r);

}
