static void spapr_populate_pci_devices_dt(PCIBus *bus, PCIDevice *pdev,

                                          void *opaque)

{

    PCIBus *sec_bus;

    sPAPRFDT *p = opaque;

    int offset;

    sPAPRFDT s_fdt;

    uint32_t drc_index = spapr_phb_get_pci_drc_index(p->sphb, pdev);



    offset = spapr_create_pci_child_dt(p->sphb, pdev,

                                       drc_index, NULL,

                                       p->fdt, p->node_off);

    if (!offset) {

        error_report("Failed to create pci child device tree node");

        return;

    }



    if ((pci_default_read_config(pdev, PCI_HEADER_TYPE, 1) !=

         PCI_HEADER_TYPE_BRIDGE)) {

        return;

    }



    sec_bus = pci_bridge_get_sec_bus(PCI_BRIDGE(pdev));

    if (!sec_bus) {

        return;

    }



    s_fdt.fdt = p->fdt;

    s_fdt.node_off = offset;

    s_fdt.sphb = p->sphb;

    pci_for_each_device(sec_bus, pci_bus_num(sec_bus),

                        spapr_populate_pci_devices_dt,

                        &s_fdt);

}
