static int ioh3420_initfn(PCIDevice *d)

{

    PCIEPort *p = PCIE_PORT(d);

    PCIESlot *s = PCIE_SLOT(d);

    int rc;



    pci_bridge_initfn(d, TYPE_PCIE_BUS);

    pcie_port_init_reg(d);



    rc = pci_bridge_ssvid_init(d, IOH_EP_SSVID_OFFSET,

                               IOH_EP_SSVID_SVID, IOH_EP_SSVID_SSID);

    if (rc < 0) {

        goto err_bridge;

    }



    rc = msi_init(d, IOH_EP_MSI_OFFSET, IOH_EP_MSI_NR_VECTOR,

                  IOH_EP_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_64BIT,

                  IOH_EP_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_MASKBIT);

    if (rc < 0) {

        goto err_bridge;

    }



    rc = pcie_cap_init(d, IOH_EP_EXP_OFFSET, PCI_EXP_TYPE_ROOT_PORT, p->port);

    if (rc < 0) {

        goto err_msi;

    }



    pcie_cap_arifwd_init(d);

    pcie_cap_deverr_init(d);

    pcie_cap_slot_init(d, s->slot);

    pcie_cap_root_init(d);



    pcie_chassis_create(s->chassis);

    rc = pcie_chassis_add_slot(s);

    if (rc < 0) {

        goto err_pcie_cap;

    }



    rc = pcie_aer_init(d, IOH_EP_AER_OFFSET, PCI_ERR_SIZEOF);

    if (rc < 0) {

        goto err;

    }

    pcie_aer_root_init(d);

    ioh3420_aer_vector_update(d);



    return 0;



err:

    pcie_chassis_del_slot(s);

err_pcie_cap:

    pcie_cap_exit(d);

err_msi:

    msi_uninit(d);

err_bridge:

    pci_bridge_exitfn(d);

    return rc;

}
