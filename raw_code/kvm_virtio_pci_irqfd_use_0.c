static int kvm_virtio_pci_irqfd_use(VirtIOPCIProxy *proxy,

                                 unsigned int queue_no,

                                 unsigned int vector)

{

    VirtIOIRQFD *irqfd = &proxy->vector_irqfd[vector];

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtQueue *vq = virtio_get_queue(vdev, queue_no);

    EventNotifier *n = virtio_queue_get_guest_notifier(vq);

    int ret;

    ret = kvm_irqchip_add_irqfd_notifier_gsi(kvm_state, n, NULL, irqfd->virq);

    return ret;

}
