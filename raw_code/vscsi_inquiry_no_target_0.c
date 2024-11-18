static void vscsi_inquiry_no_target(VSCSIState *s, vscsi_req *req)

{

    uint8_t *cdb = req->iu.srp.cmd.cdb;

    uint8_t resp_data[36];

    int rc, len, alen;



    /* We dont do EVPD. Also check that page_code is 0 */

    if ((cdb[1] & 0x01) || (cdb[1] & 0x01) || cdb[2] != 0) {

        /* Send INVALID FIELD IN CDB */

        vscsi_makeup_sense(s, req, ILLEGAL_REQUEST, 0x24, 0);

        vscsi_send_rsp(s, req, CHECK_CONDITION, 0, 0);

        return;

    }

    alen = cdb[3];

    alen = (alen << 8) | cdb[4];

    len = MIN(alen, 36);



    /* Fake up inquiry using PQ=3 */

    memset(resp_data, 0, 36);

    resp_data[0] = 0x7f;   /* Not capable of supporting a device here */

    resp_data[2] = 0x06;   /* SPS-4 */

    resp_data[3] = 0x02;   /* Resp data format */

    resp_data[4] = 36 - 5; /* Additional length */

    resp_data[7] = 0x10;   /* Sync transfers */

    memcpy(&resp_data[16], "QEMU EMPTY      ", 16);

    memcpy(&resp_data[8], "QEMU    ", 8);



    req->writing = 0;

    vscsi_preprocess_desc(req);

    rc = vscsi_srp_transfer_data(s, req, 0, resp_data, len);

    if (rc < 0) {

        vscsi_makeup_sense(s, req, HARDWARE_ERROR, 0, 0);

        vscsi_send_rsp(s, req, CHECK_CONDITION, 0, 0);

    } else {

        vscsi_send_rsp(s, req, 0, 36 - rc, 0);

    }

}
