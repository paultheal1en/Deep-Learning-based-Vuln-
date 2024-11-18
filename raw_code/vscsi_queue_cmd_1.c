static int vscsi_queue_cmd(VSCSIState *s, vscsi_req *req)

{

    union srp_iu *srp = &req->iu.srp;

    SCSIDevice *sdev;

    int n, id, lun;



    vscsi_decode_id_lun(be64_to_cpu(srp->cmd.lun), &id, &lun);



    /* Qemu vs. linux issue with LUNs to be sorted out ... */

    sdev = (id < 8 && lun < 16) ? s->bus.devs[id] : NULL;

    if (!sdev) {

        dprintf("VSCSI: Command for id %d with no drive\n", id);

        if (srp->cmd.cdb[0] == INQUIRY) {

            vscsi_inquiry_no_target(s, req);

        } else {

            vscsi_makeup_sense(s, req, ILLEGAL_REQUEST, 0x24, 0x00);

            vscsi_send_rsp(s, req, CHECK_CONDITION, 0, 0);

        } return 1;

    }



    req->sdev = sdev;

    req->lun = lun;

    n = sdev->info->send_command(sdev, req->qtag, srp->cmd.cdb, lun);



    dprintf("VSCSI: Queued command tag 0x%x CMD 0x%x ID %d LUN %d ret: %d\n",

            req->qtag, srp->cmd.cdb[0], id, lun, n);



    if (n) {

        /* Transfer direction must be set before preprocessing the

         * descriptors

         */

        req->writing = (n < 1);



        /* Preprocess RDMA descriptors */

        vscsi_preprocess_desc(req);

    }



    /* Get transfer direction and initiate transfer */

    if (n > 0) {

        req->data_len = n;

        sdev->info->read_data(sdev, req->qtag);

    } else if (n < 0) {

        req->data_len = -n;

        sdev->info->write_data(sdev, req->qtag);

    }

    /* Don't touch req here, it may have been recycled already */



    return 0;

}
