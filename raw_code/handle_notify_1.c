static void handle_notify(EventNotifier *e)

{

    VirtIOBlockDataPlane *s = container_of(e, VirtIOBlockDataPlane,

                                           host_notifier);



    VirtQueueElement *elem;

    VirtIOBlockReq *req;

    int ret;

    MultiReqBuffer mrb = {

        .num_writes = 0,

    };



    event_notifier_test_and_clear(&s->host_notifier);

    bdrv_io_plug(s->blk->conf.bs);

    for (;;) {

        /* Disable guest->host notifies to avoid unnecessary vmexits */

        vring_disable_notification(s->vdev, &s->vring);



        for (;;) {

            ret = vring_pop(s->vdev, &s->vring, &elem);

            if (ret < 0) {

                assert(elem == NULL);

                break; /* no more requests */

            }



            trace_virtio_blk_data_plane_process_request(s, elem->out_num,

                                                        elem->in_num, elem->index);



            req = g_slice_new(VirtIOBlockReq);

            req->dev = VIRTIO_BLK(s->vdev);

            req->elem = elem;

            virtio_blk_handle_request(req, &mrb);

        }



        virtio_submit_multiwrite(s->blk->conf.bs, &mrb);



        if (likely(ret == -EAGAIN)) { /* vring emptied */

            /* Re-enable guest->host notifies and stop processing the vring.

             * But if the guest has snuck in more descriptors, keep processing.

             */

            if (vring_enable_notification(s->vdev, &s->vring)) {

                break;

            }

        } else { /* fatal error */

            break;

        }

    }

    bdrv_io_unplug(s->blk->conf.bs);

}
