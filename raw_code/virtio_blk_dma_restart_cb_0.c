static void virtio_blk_dma_restart_cb(void *opaque, int running, int reason)

{

    VirtIOBlock *s = opaque;

    VirtIOBlockReq *req = s->rq;



    if (!running)

        return;



    s->rq = NULL;



    while (req) {

        virtio_blk_handle_write(req);

        req = req->next;

    }

}
