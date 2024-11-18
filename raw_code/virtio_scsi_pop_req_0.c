static VirtIOSCSIReq *virtio_scsi_pop_req(VirtIOSCSI *s, VirtQueue *vq)

{

    VirtIOSCSIReq *req;

    req = g_malloc(sizeof(*req));

    if (!virtqueue_pop(vq, &req->elem)) {

        g_free(req);

        return NULL;

    }



    virtio_scsi_parse_req(s, vq, req);

    return req;

}
