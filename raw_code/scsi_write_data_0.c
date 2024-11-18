static int scsi_write_data(SCSIDevice *d, uint32_t tag)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, d);

    SCSIDiskReq *r;



    DPRINTF("Write data tag=0x%x\n", tag);

    r = scsi_find_request(s, tag);

    if (!r) {

        BADF("Bad write tag 0x%x\n", tag);

        scsi_command_complete(r, CHECK_CONDITION, HARDWARE_ERROR);

        return 1;

    }



    /* No data transfer may already be in progress */

    assert(r->req.aiocb == NULL);



    scsi_write_request(r);



    return 0;

}
