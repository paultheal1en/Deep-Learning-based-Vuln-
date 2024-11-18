static void pcie_pci_bridge_realize(PCIDevice *d, Error **errp)

{

    PCIBridge *br = PCI_BRIDGE(d);

    PCIEPCIBridge *pcie_br = PCIE_PCI_BRIDGE_DEV(d);

    int rc, pos;



    pci_bridge_initfn(d, TYPE_PCI_BUS);



    d->config[PCI_INTERRUPT_PIN] = 0x1;

    memory_region_init(&pcie_br->shpc_bar, OBJECT(d), "shpc-bar",

                       shpc_bar_size(d));

    rc = shpc_init(d, &br->sec_bus, &pcie_br->shpc_bar, 0, errp);

    if (rc) {

        goto error;

    }



    rc = pcie_cap_init(d, 0, PCI_EXP_TYPE_PCI_BRIDGE, 0, errp);

    if (rc < 0) {

        goto cap_error;

    }



    pos = pci_add_capability(d, PCI_CAP_ID_PM, 0, PCI_PM_SIZEOF, errp);

    if (pos < 0) {

        goto pm_error;

    }

    d->exp.pm_cap = pos;

    pci_set_word(d->config + pos + PCI_PM_PMC, 0x3);



    pcie_cap_arifwd_init(d);

    pcie_cap_deverr_init(d);



    rc = pcie_aer_init(d, PCI_ERR_VER, 0x100, PCI_ERR_SIZEOF, errp);

    if (rc < 0) {

        goto aer_error;

    }



    if (pcie_br->msi != ON_OFF_AUTO_OFF) {

        rc = msi_init(d, 0, 1, true, true, errp);

        if (rc < 0) {

            goto msi_error;

        }

    }

    pci_register_bar(d, 0, PCI_BASE_ADDRESS_SPACE_MEMORY |

                     PCI_BASE_ADDRESS_MEM_TYPE_64, &pcie_br->shpc_bar);

    return;



msi_error:

    pcie_aer_exit(d);

aer_error:

pm_error:

    pcie_cap_exit(d);

cap_error:

    shpc_free(d);

error:

    pci_bridge_exitfn(d);

}
