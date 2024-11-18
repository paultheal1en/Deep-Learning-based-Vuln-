static int kvm_virtio_pci_vq_vector_unmask(VirtIOPCIProxy *proxy,

                                        unsigned int queue_no,

                                        unsigned int vector,

                                        MSIMessage msg)

{

    VirtQueue *vq = virtio_get_queue(proxy->vdev, queue_no);

    EventNotifier *n = virtio_queue_get_guest_notifier(vq);

    VirtIOIRQFD *irqfd = &proxy->vector_irqfd[vector];

    int ret;



    if (irqfd->msg.data != msg.data || irqfd->msg.address != msg.address) {

        ret = kvm_irqchip_update_msi_route(kvm_state, irqfd->virq, msg);

        if (ret < 0) {

            return ret;

        }

    }



    /* If guest supports masking, irqfd is already setup, unmask it.

     * Otherwise, set it up now.

     */

    if (proxy->vdev->guest_notifier_mask) {

        proxy->vdev->guest_notifier_mask(proxy->vdev, queue_no, false);

        /* Test after unmasking to avoid losing events. */

        if (proxy->vdev->guest_notifier_pending &&

            proxy->vdev->guest_notifier_pending(proxy->vdev, queue_no)) {

            event_notifier_set(n);

        }

    } else {

        ret = kvm_virtio_pci_irqfd_use(proxy, queue_no, vector);

    }

    return ret;

}
