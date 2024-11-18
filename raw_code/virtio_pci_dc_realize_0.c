static void virtio_pci_dc_realize(DeviceState *qdev, Error **errp)

{

    VirtioPCIClass *vpciklass = VIRTIO_PCI_GET_CLASS(qdev);

    VirtIOPCIProxy *proxy = VIRTIO_PCI(qdev);

    PCIDevice *pci_dev = &proxy->pci_dev;



    if (!(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_PCIE) &&

        !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_MODERN)) {

        pci_dev->cap_present |= QEMU_PCI_CAP_EXPRESS;

    }



    vpciklass->parent_dc_realize(qdev, errp);

}
