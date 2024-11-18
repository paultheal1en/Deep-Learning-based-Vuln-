static void virtio_blk_handle_output(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    VirtIOBlockReq *req;

    MultiReqBuffer mrb = {};



    /* Some guests kick before setting VIRTIO_CONFIG_S_DRIVER_OK so start

     * dataplane here instead of waiting for .set_status().

     */

    if (s->dataplane) {

        virtio_blk_data_plane_start(s->dataplane);

        return;

    }



    blk_io_plug(s->blk);



    while ((req = virtio_blk_get_request(s))) {

        virtio_blk_handle_request(req, &mrb);

    }



    if (mrb.num_reqs) {

        virtio_blk_submit_multireq(s->blk, &mrb);

    }



    blk_io_unplug(s->blk);

}
