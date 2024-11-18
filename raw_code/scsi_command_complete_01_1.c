static void scsi_command_complete(void *opaque, int ret)

{

    SCSIGenericReq *r = (SCSIGenericReq *)opaque;

    SCSIGenericState *s = DO_UPCAST(SCSIGenericState, qdev, r->req.dev);



    r->req.aiocb = NULL;

    s->driver_status = r->io_header.driver_status;

    if (s->driver_status & SG_ERR_DRIVER_SENSE)

        s->senselen = r->io_header.sb_len_wr;



    if (ret != 0)

        r->req.status = BUSY;

    else {

        if (s->driver_status & SG_ERR_DRIVER_TIMEOUT) {

            r->req.status = BUSY;

            BADF("Driver Timeout\n");

        } else if (r->io_header.status)

            r->req.status = r->io_header.status;

        else if (s->driver_status & SG_ERR_DRIVER_SENSE)

            r->req.status = CHECK_CONDITION;

        else

            r->req.status = GOOD;

    }

    DPRINTF("Command complete 0x%p tag=0x%x status=%d\n",

            r, r->req.tag, r->req.status);



    scsi_req_complete(&r->req);

}
