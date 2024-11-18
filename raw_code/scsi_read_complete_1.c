static void scsi_read_complete(void * opaque, int ret)

{

    SCSIGenericReq *r = (SCSIGenericReq *)opaque;

    int len;



    if (ret) {

        DPRINTF("IO error ret %d\n", ret);

        scsi_command_complete(r, ret);

        return;

    }

    len = r->io_header.dxfer_len - r->io_header.resid;

    DPRINTF("Data ready tag=0x%x len=%d\n", r->req.tag, len);



    r->len = -1;

    r->req.bus->complete(r->req.bus, SCSI_REASON_DATA, r->req.tag, len);

    if (len == 0)

        scsi_command_complete(r, 0);

}
