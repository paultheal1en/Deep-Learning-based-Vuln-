static void virtio_blk_handle_flush(VirtIOBlockReq *req, MultiReqBuffer *mrb)

{

    block_acct_start(bdrv_get_stats(req->dev->bs), &req->acct, 0,

                     BLOCK_ACCT_FLUSH);



    /*

     * Make sure all outstanding writes are posted to the backing device.

     */

    virtio_submit_multiwrite(req->dev->bs, mrb);

    bdrv_aio_flush(req->dev->bs, virtio_blk_flush_complete, req);

}
