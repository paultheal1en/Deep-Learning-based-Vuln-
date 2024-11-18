static void spapr_phb_add_pci_device(sPAPRDRConnector *drc,

                                     sPAPRPHBState *phb,

                                     PCIDevice *pdev,

                                     Error **errp)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    DeviceState *dev = DEVICE(pdev);

    int drc_index = drck->get_index(drc);

    void *fdt = NULL;

    int fdt_start_offset = 0, fdt_size;



    if (dev->hotplugged) {

        fdt = create_device_tree(&fdt_size);

        fdt_start_offset = spapr_create_pci_child_dt(phb, pdev,

                                                     drc_index, NULL,

                                                     fdt, 0);

        if (!fdt_start_offset) {

            error_setg(errp, "Failed to create pci child device tree node");

            goto out;

        }

    }



    drck->attach(drc, DEVICE(pdev),

                 fdt, fdt_start_offset, !dev->hotplugged, errp);

out:

    if (*errp) {

        g_free(fdt);

    }

}
