int vhost_dev_enable_notifiers(struct vhost_dev *hdev, VirtIODevice *vdev)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(vdev)));

    VirtioBusState *vbus = VIRTIO_BUS(qbus);

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(vbus);

    int i, r;

    if (!k->set_host_notifier) {

        fprintf(stderr, "binding does not support host notifiers\n");

        r = -ENOSYS;

        goto fail;

    }



    for (i = 0; i < hdev->nvqs; ++i) {

        r = k->set_host_notifier(qbus->parent, hdev->vq_index + i, true);

        if (r < 0) {

            fprintf(stderr, "vhost VQ %d notifier binding failed: %d\n", i, -r);

            goto fail_vq;

        }

    }



    return 0;

fail_vq:

    while (--i >= 0) {

        r = k->set_host_notifier(qbus->parent, hdev->vq_index + i, false);

        if (r < 0) {

            fprintf(stderr, "vhost VQ %d notifier cleanup error: %d\n", i, -r);

            fflush(stderr);

        }

        assert (r >= 0);

    }

fail:

    return r;

}
