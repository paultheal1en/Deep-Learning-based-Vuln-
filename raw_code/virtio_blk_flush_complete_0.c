static void virtio_blk_flush_complete(void *opaque, int ret)

{

    VirtIOBlockReq *req = opaque;



    if (ret) {

        if (virtio_blk_handle_rw_error(req, -ret, 0)) {

            return;

        }

    }



    virtio_blk_req_complete(req, VIRTIO_BLK_S_OK);

    block_acct_done(bdrv_get_stats(req->dev->bs), &req->acct);

    virtio_blk_free_request(req);

}
