static VirtIOSCSIReq *virtio_scsi_init_req(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req;

    req = g_malloc(sizeof(*req));



    req->vq = vq;

    req->dev = s;

    req->sreq = NULL;

    qemu_sglist_init(&req->qsgl, DEVICE(s), 8, &address_space_memory);

    return req;

}
