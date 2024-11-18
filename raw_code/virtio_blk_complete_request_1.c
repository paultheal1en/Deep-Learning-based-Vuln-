static void virtio_blk_complete_request(VirtIOBlockReq *req,

                                        unsigned char status)

{

    VirtIOBlock *s = req->dev;

    VirtIODevice *vdev = VIRTIO_DEVICE(s);



    trace_virtio_blk_req_complete(req, status);



    stb_p(&req->in->status, status);

    virtqueue_push(s->vq, req->elem, req->qiov.size + sizeof(*req->in));

    virtio_notify(vdev, s->vq);

}
