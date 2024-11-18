bool vfio_blacklist_opt_rom(VFIOPCIDevice *vdev)

{

    PCIDevice *pdev = &vdev->pdev;

    uint16_t vendor_id, device_id;

    int count = 0;



    vendor_id = pci_get_word(pdev->config + PCI_VENDOR_ID);

    device_id = pci_get_word(pdev->config + PCI_DEVICE_ID);



    while (count < ARRAY_SIZE(romblacklist)) {

        if (romblacklist[count].vendor_id == vendor_id &&

            romblacklist[count].device_id == device_id) {

                return true;

        }

        count++;

    }



    return false;

}
