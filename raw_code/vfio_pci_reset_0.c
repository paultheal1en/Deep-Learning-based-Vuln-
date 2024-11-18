static void vfio_pci_reset(DeviceState *dev)

{

    PCIDevice *pdev = DO_UPCAST(PCIDevice, qdev, dev);

    VFIODevice *vdev = DO_UPCAST(VFIODevice, pdev, pdev);



    if (!vdev->reset_works) {

        return;

    }



    if (ioctl(vdev->fd, VFIO_DEVICE_RESET)) {

        error_report("vfio: Error unable to reset physical device "

                     "(%04x:%02x:%02x.%x): %m\n", vdev->host.domain,

                     vdev->host.bus, vdev->host.slot, vdev->host.function);

    }

}
