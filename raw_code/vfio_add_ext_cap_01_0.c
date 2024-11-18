static void vfio_add_ext_cap(VFIOPCIDevice *vdev)

{

    PCIDevice *pdev = &vdev->pdev;

    uint32_t header;

    uint16_t cap_id, next, size;

    uint8_t cap_ver;

    uint8_t *config;



    /* Only add extended caps if we have them and the guest can see them */

    if (!pci_is_express(pdev) || !pci_bus_is_express(pdev->bus) ||

        !pci_get_long(pdev->config + PCI_CONFIG_SPACE_SIZE)) {

        return;

    }



    /*

     * pcie_add_capability always inserts the new capability at the tail

     * of the chain.  Therefore to end up with a chain that matches the

     * physical device, we cache the config space to avoid overwriting

     * the original config space when we parse the extended capabilities.

     */

    config = g_memdup(pdev->config, vdev->config_size);



    /*

     * Extended capabilities are chained with each pointing to the next, so we

     * can drop anything other than the head of the chain simply by modifying

     * the previous next pointer.  For the head of the chain, we can modify the

     * capability ID to something that cannot match a valid capability.  ID

     * 0 is reserved for this since absence of capabilities is indicated by

     * 0 for the ID, version, AND next pointer.  However, pcie_add_capability()

     * uses ID 0 as reserved for list management and will incorrectly match and

     * assert if we attempt to pre-load the head of the chain with this ID.

     * Use ID 0xFFFF temporarily since it is also seems to be reserved in

     * part for identifying absence of capabilities in a root complex register

     * block.  If the ID still exists after adding capabilities, switch back to

     * zero.  We'll mark this entire first dword as emulated for this purpose.

     */

    pci_set_long(pdev->config + PCI_CONFIG_SPACE_SIZE,

                 PCI_EXT_CAP(0xFFFF, 0, 0));

    pci_set_long(pdev->wmask + PCI_CONFIG_SPACE_SIZE, 0);

    pci_set_long(vdev->emulated_config_bits + PCI_CONFIG_SPACE_SIZE, ~0);



    for (next = PCI_CONFIG_SPACE_SIZE; next;

         next = PCI_EXT_CAP_NEXT(pci_get_long(config + next))) {

        header = pci_get_long(config + next);

        cap_id = PCI_EXT_CAP_ID(header);

        cap_ver = PCI_EXT_CAP_VER(header);



        /*

         * If it becomes important to configure extended capabilities to their

         * actual size, use this as the default when it's something we don't

         * recognize. Since QEMU doesn't actually handle many of the config

         * accesses, exact size doesn't seem worthwhile.

         */

        size = vfio_ext_cap_max_size(config, next);



        /* Use emulated next pointer to allow dropping extended caps */

        pci_long_test_and_set_mask(vdev->emulated_config_bits + next,

                                   PCI_EXT_CAP_NEXT_MASK);



        switch (cap_id) {

        case PCI_EXT_CAP_ID_SRIOV: /* Read-only VF BARs confuse OVMF */

        case PCI_EXT_CAP_ID_ARI: /* XXX Needs next function virtualization */

            trace_vfio_add_ext_cap_dropped(vdev->vbasedev.name, cap_id, next);

            break;

        default:

            pcie_add_capability(pdev, cap_id, cap_ver, next, size);

        }



    }



    /* Cleanup chain head ID if necessary */

    if (pci_get_word(pdev->config + PCI_CONFIG_SPACE_SIZE) == 0xFFFF) {

        pci_set_word(pdev->config + PCI_CONFIG_SPACE_SIZE, 0);

    }



    g_free(config);

    return;

}
