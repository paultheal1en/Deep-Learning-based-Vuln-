static int xio3130_downstream_initfn(PCIDevice *d)

{

    PCIBridge* br = DO_UPCAST(PCIBridge, dev, d);

    PCIEPort *p = DO_UPCAST(PCIEPort, br, br);

    PCIESlot *s = DO_UPCAST(PCIESlot, port, p);

    int rc;

    int tmp;



    rc = pci_bridge_initfn(d);

    if (rc < 0) {

        return rc;

    }



    pcie_port_init_reg(d);

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_TI);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_TI_XIO3130D);

    d->config[PCI_REVISION_ID] = XIO3130_REVISION;



    rc = msi_init(d, XIO3130_MSI_OFFSET, XIO3130_MSI_NR_VECTOR,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_64BIT,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_MASKBIT);

    if (rc < 0) {

        goto err_bridge;

    }

    rc = pci_bridge_ssvid_init(d, XIO3130_SSVID_OFFSET,

                               XIO3130_SSVID_SVID, XIO3130_SSVID_SSID);

    if (rc < 0) {

        goto err_bridge;

    }

    rc = pcie_cap_init(d, XIO3130_EXP_OFFSET, PCI_EXP_TYPE_DOWNSTREAM,

                       p->port);

    if (rc < 0) {

        goto err_msi;

    }

    pcie_cap_flr_init(d);

    pcie_cap_deverr_init(d);

    pcie_cap_slot_init(d, s->slot);

    pcie_chassis_create(s->chassis);

    rc = pcie_chassis_add_slot(s);

    if (rc < 0) {

        goto err_pcie_cap;

    }

    pcie_cap_ari_init(d);

    rc = pcie_aer_init(d, XIO3130_AER_OFFSET);

    if (rc < 0) {

        goto err;

    }



    return 0;



err:

    pcie_chassis_del_slot(s);

err_pcie_cap:

    pcie_cap_exit(d);

err_msi:

    msi_uninit(d);

err_bridge:

    tmp = pci_bridge_exitfn(d);

    assert(!tmp);

    return rc;

}
