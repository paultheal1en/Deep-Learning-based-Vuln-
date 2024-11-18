static int kvm_virtio_pci_vq_vector_use(VirtIOPCIProxy *proxy,

                                        unsigned int queue_no,

                                        unsigned int vector,

                                        MSIMessage msg)

{

    VirtQueue *vq = virtio_get_queue(proxy->vdev, queue_no);

    EventNotifier *n = virtio_queue_get_guest_notifier(vq);

    VirtIOIRQFD *irqfd = &proxy->vector_irqfd[vector];

    int ret;



    if (irqfd->users == 0) {

        ret = kvm_irqchip_add_msi_route(kvm_state, msg);

        if (ret < 0) {

            return ret;

        }

        irqfd->virq = ret;

    }

    irqfd->users++;



    ret = kvm_irqchip_add_irq_notifier(kvm_state, n, irqfd->virq);

    if (ret < 0) {

        if (--irqfd->users == 0) {

            kvm_irqchip_release_virq(kvm_state, irqfd->virq);

        }

        return ret;

    }



    virtio_queue_set_guest_notifier_fd_handler(vq, true, true);

    return 0;

}
