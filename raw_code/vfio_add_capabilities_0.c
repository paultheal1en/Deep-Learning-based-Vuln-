static int vfio_add_capabilities(VFIOPCIDevice *vdev)

{

    PCIDevice *pdev = &vdev->pdev;

    int ret;



    if (!(pdev->config[PCI_STATUS] & PCI_STATUS_CAP_LIST) ||

        !pdev->config[PCI_CAPABILITY_LIST]) {

        return 0; /* Nothing to add */

    }



    ret = vfio_add_std_cap(vdev, pdev->config[PCI_CAPABILITY_LIST]);

    if (ret) {

        return ret;

    }



    /* on PCI bus, it doesn't make sense to expose extended capabilities. */

    if (!pci_is_express(pdev) ||

        !pci_bus_is_express(pdev->bus) ||

        !pci_get_long(pdev->config + PCI_CONFIG_SPACE_SIZE)) {

        return 0;

    }



    return vfio_add_ext_cap(vdev);

}
