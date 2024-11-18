static void virtio_pci_vmstate_change(DeviceState *d, bool running)

{

    VirtIOPCIProxy *proxy = to_virtio_pci_proxy(d);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);



    if (running) {

        /* Try to find out if the guest has bus master disabled, but is

           in ready state. Then we have a buggy guest OS. */

        if ((vdev->status & VIRTIO_CONFIG_S_DRIVER_OK) &&

            !(proxy->pci_dev.config[PCI_COMMAND] & PCI_COMMAND_MASTER)) {

            proxy->flags |= VIRTIO_PCI_FLAG_BUS_MASTER_BUG;

        }

        virtio_pci_start_ioeventfd(proxy);

    } else {

        virtio_pci_stop_ioeventfd(proxy);

    }

}
