static int vhost_virtqueue_init(struct vhost_dev *dev,

                                struct vhost_virtqueue *vq, int n)

{

    struct vhost_vring_file file = {

        .index = n,

    };

    int r = event_notifier_init(&vq->masked_notifier, 0);

    if (r < 0) {

        return r;

    }



    file.fd = event_notifier_get_fd(&vq->masked_notifier);

    r = dev->vhost_ops->vhost_call(dev, VHOST_SET_VRING_CALL, &file);

    if (r) {

        r = -errno;

        goto fail_call;

    }

    return 0;

fail_call:

    event_notifier_cleanup(&vq->masked_notifier);

    return r;

}
