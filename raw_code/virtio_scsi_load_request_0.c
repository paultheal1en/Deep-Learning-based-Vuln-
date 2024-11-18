static void *virtio_scsi_load_request(QEMUFile *f, SCSIRequest *sreq)

{

    SCSIBus *bus = sreq->bus;

    VirtIOSCSI *s = container_of(bus, VirtIOSCSI, bus);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    VirtIOSCSIReq *req;

    uint32_t n;



    req = g_malloc(sizeof(*req));

    qemu_get_be32s(f, &n);

    assert(n < vs->conf.num_queues);

    qemu_get_buffer(f, (unsigned char *)&req->elem, sizeof(req->elem));

    /* TODO: add a way for SCSIBusInfo's load_request to fail,

     * and fail migration instead of asserting here.

     * When we do, we might be able to re-enable NDEBUG below.

     */

#ifdef NDEBUG

#error building with NDEBUG is not supported

#endif

    assert(req->elem.in_num <= ARRAY_SIZE(req->elem.in_sg));

    assert(req->elem.out_num <= ARRAY_SIZE(req->elem.out_sg));

    virtio_scsi_parse_req(s, vs->cmd_vqs[n], req);



    scsi_req_ref(sreq);

    req->sreq = sreq;

    if (req->sreq->cmd.mode != SCSI_XFER_NONE) {

        int req_mode =

            (req->elem.in_num > 1 ? SCSI_XFER_FROM_DEV : SCSI_XFER_TO_DEV);



        assert(req->sreq->cmd.mode == req_mode);

    }

    return req;

}
