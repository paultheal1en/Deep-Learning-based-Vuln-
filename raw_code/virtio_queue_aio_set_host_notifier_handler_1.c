void virtio_queue_aio_set_host_notifier_handler(VirtQueue *vq, AioContext *ctx,

                                                bool assign, bool set_handler)

{

    if (assign && set_handler) {

        aio_set_event_notifier(ctx, &vq->host_notifier, true,

                               virtio_queue_host_notifier_read);

    } else {

        aio_set_event_notifier(ctx, &vq->host_notifier, true, NULL);

    }

    if (!assign) {

        /* Test and clear notifier before after disabling event,

         * in case poll callback didn't have time to run. */

        virtio_queue_host_notifier_read(&vq->host_notifier);

    }

}
