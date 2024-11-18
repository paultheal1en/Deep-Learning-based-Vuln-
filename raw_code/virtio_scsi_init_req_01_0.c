VirtIOSCSIReq *virtio_scsi_init_req(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req;

    VirtIOSCSICommon *vs = (VirtIOSCSICommon *)s;

    const size_t zero_skip = offsetof(VirtIOSCSIReq, vring);



    req = g_malloc(sizeof(*req) + vs->cdb_size);

    req->vq = vq;

    req->dev = s;

    qemu_sglist_init(&req->qsgl, DEVICE(s), 8, &address_space_memory);

    qemu_iovec_init(&req->resp_iov, 1);

    memset((uint8_t *)req + zero_skip, 0, sizeof(*req) - zero_skip);

    return req;

}
