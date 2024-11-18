static int vfio_setup_pcie_cap(VFIOPCIDevice *vdev, int pos, uint8_t size,
                               Error **errp)
{
    uint16_t flags;
    uint8_t type;
    flags = pci_get_word(vdev->pdev.config + pos + PCI_CAP_FLAGS);
    type = (flags & PCI_EXP_FLAGS_TYPE) >> 4;
    if (type != PCI_EXP_TYPE_ENDPOINT &&
        type != PCI_EXP_TYPE_LEG_END &&
        type != PCI_EXP_TYPE_RC_END) {
        error_setg(errp, "assignment of PCIe type 0x%x "
                   "devices is not currently supported", type);
        return -EINVAL;
    if (!pci_bus_is_express(vdev->pdev.bus)) {
        PCIBus *bus = vdev->pdev.bus;
        PCIDevice *bridge;
         * Traditionally PCI device assignment exposes the PCIe capability
         * as-is on non-express buses.  The reason being that some drivers
         * simply assume that it's there, for example tg3.  However when
         * we're running on a native PCIe machine type, like Q35, we need
         * to hide the PCIe capability.  The reason for this is twofold;
         * first Windows guests get a Code 10 error when the PCIe capability
         * is exposed in this configuration.  Therefore express devices won't
         * work at all unless they're attached to express buses in the VM.
         * Second, a native PCIe machine introduces the possibility of fine
         * granularity IOMMUs supporting both translation and isolation.
         * Guest code to discover the IOMMU visibility of a device, such as
         * IOMMU grouping code on Linux, is very aware of device types and
         * valid transitions between bus types.  An express device on a non-
         * express bus is not a valid combination on bare metal systems.
         *
         * Drivers that require a PCIe capability to make the device
         * functional are simply going to need to have their devices placed
         * on a PCIe bus in the VM.
        while (!pci_bus_is_root(bus)) {
            bridge = pci_bridge_get_device(bus);
            bus = bridge->bus;
        if (pci_bus_is_express(bus)) {
            return 0;
    } else if (pci_bus_is_root(vdev->pdev.bus)) {
         * On a Root Complex bus Endpoints become Root Complex Integrated
         * Endpoints, which changes the type and clears the LNK & LNK2 fields.
        if (type == PCI_EXP_TYPE_ENDPOINT) {
                                   PCI_EXP_TYPE_RC_END << 4,
                                   PCI_EXP_FLAGS_TYPE);
            /* Link Capabilities, Status, and Control goes away */
            if (size > PCI_EXP_LNKCTL) {
                vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP, 0, ~0);
                vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL, 0, ~0);
                vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA, 0, ~0);
#ifndef PCI_EXP_LNKCAP2
#define PCI_EXP_LNKCAP2 44
#endif
#ifndef PCI_EXP_LNKSTA2
#define PCI_EXP_LNKSTA2 50
#endif
                /* Link 2 Capabilities, Status, and Control goes away */
                if (size > PCI_EXP_LNKCAP2) {
                    vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP2, 0, ~0);
                    vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL2, 0, ~0);
                    vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA2, 0, ~0);
        } else if (type == PCI_EXP_TYPE_LEG_END) {
             * Legacy endpoints don't belong on the root complex.  Windows
             * seems to be happier with devices if we skip the capability.
            return 0;
    } else {
         * Convert Root Complex Integrated Endpoints to regular endpoints.
         * These devices don't support LNK/LNK2 capabilities, so make them up.
        if (type == PCI_EXP_TYPE_RC_END) {
                                   PCI_EXP_TYPE_ENDPOINT << 4,
                                   PCI_EXP_FLAGS_TYPE);
            vfio_add_emulated_long(vdev, pos + PCI_EXP_LNKCAP,
                                   PCI_EXP_LNK_MLW_1 | PCI_EXP_LNK_LS_25, ~0);
            vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKCTL, 0, ~0);
        /* Mark the Link Status bits as emulated to allow virtual negotiation */
        vfio_add_emulated_word(vdev, pos + PCI_EXP_LNKSTA,
                               pci_get_word(vdev->pdev.config + pos +
                                            PCI_EXP_LNKSTA),
                               PCI_EXP_LNKCAP_MLW | PCI_EXP_LNKCAP_SLS);
    pos = pci_add_capability(&vdev->pdev, PCI_CAP_ID_EXP, pos, size,
                             errp);
    if (pos < 0) {
        return pos;
    vdev->pdev.exp.exp_cap = pos;
    return pos;