static void virtio_ccw_stop_ioeventfd(VirtioCcwDevice *dev)

{

    VirtIODevice *vdev;

    int n, r;



    if (!dev->ioeventfd_started) {

        return;

    }

    vdev = virtio_bus_get_device(&dev->bus);

    for (n = 0; n < VIRTIO_PCI_QUEUE_MAX; n++) {

        if (!virtio_queue_get_num(vdev, n)) {

            continue;

        }

        r = virtio_ccw_set_guest2host_notifier(dev, n, false, false);

        assert(r >= 0);

    }

    dev->ioeventfd_started = false;

}
