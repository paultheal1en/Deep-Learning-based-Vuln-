static void spapr_phb_hot_unplug_child(HotplugHandler *plug_handler,

                                       DeviceState *plugged_dev, Error **errp)

{

    sPAPRPHBState *phb = SPAPR_PCI_HOST_BRIDGE(DEVICE(plug_handler));

    PCIDevice *pdev = PCI_DEVICE(plugged_dev);

    sPAPRDRConnectorClass *drck;

    sPAPRDRConnector *drc = spapr_phb_get_pci_drc(phb, pdev);

    Error *local_err = NULL;



    if (!phb->dr_enabled) {

        error_setg(errp, QERR_BUS_NO_HOTPLUG,

                   object_get_typename(OBJECT(phb)));

        return;

    }



    g_assert(drc);



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    if (!drck->release_pending(drc)) {

        spapr_phb_remove_pci_device(drc, phb, pdev, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

        spapr_hotplug_req_remove_by_index(drc);

    }

}
