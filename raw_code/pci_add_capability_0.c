int pci_add_capability(PCIDevice *pdev, uint8_t cap_id,

                       uint8_t offset, uint8_t size,

                       Error **errp)

{

    uint8_t *config;

    int i, overlapping_cap;



    if (!offset) {

        offset = pci_find_space(pdev, size);

        /* out of PCI config space is programming error */

        assert(offset);

    } else {

        /* Verify that capabilities don't overlap.  Note: device assignment

         * depends on this check to verify that the device is not broken.

         * Should never trigger for emulated devices, but it's helpful

         * for debugging these. */

        for (i = offset; i < offset + size; i++) {

            overlapping_cap = pci_find_capability_at_offset(pdev, i);

            if (overlapping_cap) {

                error_setg(errp, "%s:%02x:%02x.%x "

                           "Attempt to add PCI capability %x at offset "

                           "%x overlaps existing capability %x at offset %x",

                           pci_root_bus_path(pdev), pci_bus_num(pdev->bus),

                           PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn),

                           cap_id, offset, overlapping_cap, i);

                return -EINVAL;

            }

        }

    }



    config = pdev->config + offset;

    config[PCI_CAP_LIST_ID] = cap_id;

    config[PCI_CAP_LIST_NEXT] = pdev->config[PCI_CAPABILITY_LIST];

    pdev->config[PCI_CAPABILITY_LIST] = offset;

    pdev->config[PCI_STATUS] |= PCI_STATUS_CAP_LIST;

    memset(pdev->used + offset, 0xFF, QEMU_ALIGN_UP(size, 4));

    /* Make capability read-only by default */

    memset(pdev->wmask + offset, 0, size);

    /* Check capability by default */

    memset(pdev->cmask + offset, 0xFF, size);

    return offset;

}
