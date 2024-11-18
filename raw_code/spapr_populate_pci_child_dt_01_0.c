static int spapr_populate_pci_child_dt(PCIDevice *dev, void *fdt, int offset,

                                       int phb_index, int drc_index,

                                       sPAPRPHBState *sphb)

{

    ResourceProps rp;

    bool is_bridge = false;

    int pci_status, err;

    char *buf = NULL;



    if (pci_default_read_config(dev, PCI_HEADER_TYPE, 1) ==

        PCI_HEADER_TYPE_BRIDGE) {

        is_bridge = true;

    }



    /* in accordance with PAPR+ v2.7 13.6.3, Table 181 */

    _FDT(fdt_setprop_cell(fdt, offset, "vendor-id",

                          pci_default_read_config(dev, PCI_VENDOR_ID, 2)));

    _FDT(fdt_setprop_cell(fdt, offset, "device-id",

                          pci_default_read_config(dev, PCI_DEVICE_ID, 2)));

    _FDT(fdt_setprop_cell(fdt, offset, "revision-id",

                          pci_default_read_config(dev, PCI_REVISION_ID, 1)));

    _FDT(fdt_setprop_cell(fdt, offset, "class-code",

                          pci_default_read_config(dev, PCI_CLASS_PROG, 3)));

    if (pci_default_read_config(dev, PCI_INTERRUPT_PIN, 1)) {

        _FDT(fdt_setprop_cell(fdt, offset, "interrupts",

                 pci_default_read_config(dev, PCI_INTERRUPT_PIN, 1)));

    }



    if (!is_bridge) {

        _FDT(fdt_setprop_cell(fdt, offset, "min-grant",

            pci_default_read_config(dev, PCI_MIN_GNT, 1)));

        _FDT(fdt_setprop_cell(fdt, offset, "max-latency",

            pci_default_read_config(dev, PCI_MAX_LAT, 1)));

    }



    if (pci_default_read_config(dev, PCI_SUBSYSTEM_ID, 2)) {

        _FDT(fdt_setprop_cell(fdt, offset, "subsystem-id",

                 pci_default_read_config(dev, PCI_SUBSYSTEM_ID, 2)));

    }



    if (pci_default_read_config(dev, PCI_SUBSYSTEM_VENDOR_ID, 2)) {

        _FDT(fdt_setprop_cell(fdt, offset, "subsystem-vendor-id",

                 pci_default_read_config(dev, PCI_SUBSYSTEM_VENDOR_ID, 2)));

    }



    _FDT(fdt_setprop_cell(fdt, offset, "cache-line-size",

        pci_default_read_config(dev, PCI_CACHE_LINE_SIZE, 1)));



    /* the following fdt cells are masked off the pci status register */

    pci_status = pci_default_read_config(dev, PCI_STATUS, 2);

    _FDT(fdt_setprop_cell(fdt, offset, "devsel-speed",

                          PCI_STATUS_DEVSEL_MASK & pci_status));



    if (pci_status & PCI_STATUS_FAST_BACK) {

        _FDT(fdt_setprop(fdt, offset, "fast-back-to-back", NULL, 0));

    }

    if (pci_status & PCI_STATUS_66MHZ) {

        _FDT(fdt_setprop(fdt, offset, "66mhz-capable", NULL, 0));

    }

    if (pci_status & PCI_STATUS_UDF) {

        _FDT(fdt_setprop(fdt, offset, "udf-supported", NULL, 0));

    }



    /* NOTE: this is normally generated by firmware via path/unit name,

     * but in our case we must set it manually since it does not get

     * processed by OF beforehand

     */

    _FDT(fdt_setprop_string(fdt, offset, "name", "pci"));

    buf = spapr_phb_get_loc_code(sphb, dev);

    if (!buf) {

        error_report("Failed setting the ibm,loc-code");

        return -1;

    }



    err = fdt_setprop_string(fdt, offset, "ibm,loc-code", buf);

    g_free(buf);

    if (err < 0) {

        return err;

    }



    _FDT(fdt_setprop_cell(fdt, offset, "ibm,my-drc-index", drc_index));



    _FDT(fdt_setprop_cell(fdt, offset, "#address-cells",

                          RESOURCE_CELLS_ADDRESS));

    _FDT(fdt_setprop_cell(fdt, offset, "#size-cells",

                          RESOURCE_CELLS_SIZE));

    _FDT(fdt_setprop_cell(fdt, offset, "ibm,req#msi-x",

                          RESOURCE_CELLS_SIZE));



    populate_resource_props(dev, &rp);

    _FDT(fdt_setprop(fdt, offset, "reg", (uint8_t *)rp.reg, rp.reg_len));

    _FDT(fdt_setprop(fdt, offset, "assigned-addresses",

                     (uint8_t *)rp.assigned, rp.assigned_len));



    return 0;

}
