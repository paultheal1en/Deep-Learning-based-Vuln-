static int virtio_pci_stop_ioeventfd(VirtIOPCIProxy *proxy)

{

    int n;



    if (!proxy->ioeventfd_started) {

        return 0;

    }



    for (n = 0; n < VIRTIO_PCI_QUEUE_MAX; n++) {

        if (!virtio_queue_get_num(proxy->vdev, n)) {

            continue;

        }



        virtio_pci_set_host_notifier_fd_handler(proxy, n, false);

        virtio_pci_set_host_notifier_internal(proxy, n, false);

    }

    proxy->ioeventfd_started = false;

    return 0;

}
