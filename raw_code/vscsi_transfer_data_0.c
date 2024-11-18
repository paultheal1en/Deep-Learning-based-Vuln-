static void vscsi_transfer_data(SCSIRequest *sreq, uint32_t len)

{

    VSCSIState *s = VIO_SPAPR_VSCSI_DEVICE(sreq->bus->qbus.parent);

    vscsi_req *req = sreq->hba_private;

    uint8_t *buf;

    int rc = 0;



    DPRINTF("VSCSI: SCSI xfer complete tag=0x%x len=0x%x, req=%p\n",

            sreq->tag, len, req);

    if (req == NULL) {

        fprintf(stderr, "VSCSI: Can't find request for tag 0x%x\n", sreq->tag);

        return;

    }



    if (len) {

        buf = scsi_req_get_buf(sreq);

        rc = vscsi_srp_transfer_data(s, req, req->writing, buf, len);

    }

    if (rc < 0) {

        fprintf(stderr, "VSCSI: RDMA error rc=%d!\n", rc);

        vscsi_makeup_sense(s, req, HARDWARE_ERROR, 0, 0);

        scsi_req_abort(req->sreq, CHECK_CONDITION);

        return;

    }



    /* Start next chunk */

    req->data_len -= rc;

    scsi_req_continue(sreq);

}
