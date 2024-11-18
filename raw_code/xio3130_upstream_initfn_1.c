static int xio3130_upstream_initfn(PCIDevice *d)

{

    PCIEPort *p = PCIE_PORT(d);

    int rc;

    Error *err = NULL;



    pci_bridge_initfn(d, TYPE_PCIE_BUS);

    pcie_port_init_reg(d);



    rc = msi_init(d, XIO3130_MSI_OFFSET, XIO3130_MSI_NR_VECTOR,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_64BIT,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_MASKBIT, &err);

    if (rc < 0) {

        assert(rc == -ENOTSUP);

        error_report_err(err);

        goto err_bridge;

    }



    rc = pci_bridge_ssvid_init(d, XIO3130_SSVID_OFFSET,

                               XIO3130_SSVID_SVID, XIO3130_SSVID_SSID);

    if (rc < 0) {

        goto err_bridge;

    }



    rc = pcie_cap_init(d, XIO3130_EXP_OFFSET, PCI_EXP_TYPE_UPSTREAM,

                       p->port);

    if (rc < 0) {

        goto err_msi;

    }

    pcie_cap_flr_init(d);

    pcie_cap_deverr_init(d);



    rc = pcie_aer_init(d, XIO3130_AER_OFFSET, PCI_ERR_SIZEOF);

    if (rc < 0) {

        goto err;

    }



    return 0;



err:

    pcie_cap_exit(d);

err_msi:

    msi_uninit(d);

err_bridge:

    pci_bridge_exitfn(d);

    return rc;

}
