static void scsi_disk_emulate_write_data(SCSIRequest *req)

{

    SCSIDiskReq *r = DO_UPCAST(SCSIDiskReq, req, req);



    if (r->iov.iov_len) {

        int buflen = r->iov.iov_len;

        DPRINTF("Write buf_len=%zd\n", buflen);

        r->iov.iov_len = 0;

        scsi_req_data(&r->req, buflen);

        return;

    }



    switch (req->cmd.buf[0]) {

    case MODE_SELECT:

    case MODE_SELECT_10:

        /* This also clears the sense buffer for REQUEST SENSE.  */

        scsi_req_complete(&r->req, GOOD);

        break;



    default:

        abort();

    }

}
