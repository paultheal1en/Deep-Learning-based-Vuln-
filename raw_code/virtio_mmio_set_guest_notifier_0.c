static int virtio_mmio_set_guest_notifier(DeviceState *d, int n, bool assign,

                                          bool with_irqfd)

{

    VirtIOMMIOProxy *proxy = VIRTIO_MMIO(d);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);

    VirtQueue *vq = virtio_get_queue(vdev, n);

    EventNotifier *notifier = virtio_queue_get_guest_notifier(vq);



    if (assign) {

        int r = event_notifier_init(notifier, 0);

        if (r < 0) {

            return r;

        }

        virtio_queue_set_guest_notifier_fd_handler(vq, true, with_irqfd);

    } else {

        virtio_queue_set_guest_notifier_fd_handler(vq, false, with_irqfd);

        event_notifier_cleanup(notifier);

    }



    if (vdc->guest_notifier_mask) {

        vdc->guest_notifier_mask(vdev, n, !assign);

    }



    return 0;

}
