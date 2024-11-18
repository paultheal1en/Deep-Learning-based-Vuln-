static void virtio_blk_rw_complete(void *opaque, int ret)

{

    VirtIOBlockReq *next = opaque;



    while (next) {

        VirtIOBlockReq *req = next;

        next = req->mr_next;

        trace_virtio_blk_rw_complete(req, ret);



        if (req->qiov.nalloc != -1) {

            /* If nalloc is != 1 req->qiov is a local copy of the original

             * external iovec. It was allocated in submit_merged_requests

             * to be able to merge requests. */

            qemu_iovec_destroy(&req->qiov);

        }



        if (ret) {

            int p = virtio_ldl_p(VIRTIO_DEVICE(req->dev), &req->out.type);

            bool is_read = !(p & VIRTIO_BLK_T_OUT);

            /* Note that memory may be dirtied on read failure.  If the

             * virtio request is not completed here, as is the case for

             * BLOCK_ERROR_ACTION_STOP, the memory may not be copied

             * correctly during live migration.  While this is ugly,

             * it is acceptable because the device is free to write to

             * the memory until the request is completed (which will

             * happen on the other side of the migration).


            if (virtio_blk_handle_rw_error(req, -ret, is_read)) {





                continue;

            }

        }



        virtio_blk_req_complete(req, VIRTIO_BLK_S_OK);

        block_acct_done(blk_get_stats(req->dev->blk), &req->acct);

        virtio_blk_free_request(req);

    }

}