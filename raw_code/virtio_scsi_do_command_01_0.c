static uint8_t virtio_scsi_do_command(QVirtIOSCSI *vs, const uint8_t *cdb,

                                      const uint8_t *data_in,

                                      size_t data_in_len,

                                      uint8_t *data_out, size_t data_out_len)

{

    QVirtQueue *vq;

    QVirtIOSCSICmdReq req = { { 0 } };

    QVirtIOSCSICmdResp resp = { .response = 0xff, .status = 0xff };

    uint64_t req_addr, resp_addr, data_in_addr = 0, data_out_addr = 0;

    uint8_t response;

    uint32_t free_head;



    vq = vs->vq[2];



    req.lun[0] = 1; /* Select LUN */

    req.lun[1] = 1; /* Select target 1 */

    memcpy(req.cdb, cdb, CDB_SIZE);



    /* XXX: Fix endian if any multi-byte field in req/resp is used */



    /* Add request header */

    req_addr = qvirtio_scsi_alloc(vs, sizeof(req), &req);

    free_head = qvirtqueue_add(vq, req_addr, sizeof(req), false, true);



    if (data_out_len) {

        data_out_addr = qvirtio_scsi_alloc(vs, data_out_len, data_out);

        qvirtqueue_add(vq, data_out_addr, data_out_len, false, true);

    }



    /* Add response header */

    resp_addr = qvirtio_scsi_alloc(vs, sizeof(resp), &resp);

    qvirtqueue_add(vq, resp_addr, sizeof(resp), true, !!data_in_len);



    if (data_in_len) {

        data_in_addr = qvirtio_scsi_alloc(vs, data_in_len, data_in);

        qvirtqueue_add(vq, data_in_addr, data_in_len, true, false);

    }



    qvirtqueue_kick(&qvirtio_pci, vs->dev, vq, free_head);

    qvirtio_wait_queue_isr(&qvirtio_pci, vs->dev, vq, QVIRTIO_SCSI_TIMEOUT_US);



    response = readb(resp_addr + offsetof(QVirtIOSCSICmdResp, response));



    guest_free(vs->alloc, req_addr);

    guest_free(vs->alloc, resp_addr);

    guest_free(vs->alloc, data_in_addr);

    guest_free(vs->alloc, data_out_addr);

    return response;

}
