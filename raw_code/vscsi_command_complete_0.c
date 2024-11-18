static void vscsi_command_complete(SCSIRequest *sreq, uint32_t status)

{

    VSCSIState *s = DO_UPCAST(VSCSIState, vdev.qdev, sreq->bus->qbus.parent);

    vscsi_req *req = sreq->hba_private;

    int32_t res_in = 0, res_out = 0;



    dprintf("VSCSI: SCSI cmd complete, r=0x%x tag=0x%x status=0x%x, req=%p\n",

            reason, sreq->tag, status, req);

    if (req == NULL) {

        fprintf(stderr, "VSCSI: Can't find request for tag 0x%x\n", sreq->tag);

        return;

    }



    if (status == CHECK_CONDITION) {

        req->senselen = scsi_req_get_sense(req->sreq, req->sense,

                                           sizeof(req->sense));

        status = 0;

        dprintf("VSCSI: Sense data, %d bytes:\n", len);

        dprintf("       %02x  %02x  %02x  %02x  %02x  %02x  %02x  %02x\n",

                req->sense[0], req->sense[1], req->sense[2], req->sense[3],

                req->sense[4], req->sense[5], req->sense[6], req->sense[7]);

        dprintf("       %02x  %02x  %02x  %02x  %02x  %02x  %02x  %02x\n",

                req->sense[8], req->sense[9], req->sense[10], req->sense[11],

                req->sense[12], req->sense[13], req->sense[14], req->sense[15]);

    }



    dprintf("VSCSI: Command complete err=%d\n", status);

    if (status == 0) {

        /* We handle overflows, not underflows for normal commands,

         * but hopefully nobody cares

         */

        if (req->writing) {

            res_out = req->data_len;

        } else {

            res_in = req->data_len;

        }

    }

    vscsi_send_rsp(s, req, status, res_in, res_out);

    vscsi_put_req(req);

}
