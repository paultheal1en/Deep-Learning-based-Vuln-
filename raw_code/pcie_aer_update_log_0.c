static void pcie_aer_update_log(PCIDevice *dev, const PCIEAERErr *err)

{

    uint8_t *aer_cap = dev->config + dev->exp.aer_cap;

    uint8_t first_bit = ffs(err->status) - 1;

    uint32_t errcap = pci_get_long(aer_cap + PCI_ERR_CAP);

    int i;



    assert(err->status);

    assert(!(err->status & (err->status - 1)));



    errcap &= ~(PCI_ERR_CAP_FEP_MASK | PCI_ERR_CAP_TLP);

    errcap |= PCI_ERR_CAP_FEP(first_bit);



    if (err->flags & PCIE_AER_ERR_HEADER_VALID) {

        for (i = 0; i < ARRAY_SIZE(err->header); ++i) {

            /* 7.10.8 Header Log Register */

            uint8_t *header_log =

                aer_cap + PCI_ERR_HEADER_LOG + i * sizeof err->header[0];

            stl_be_p(header_log, err->header[i]);

        }

    } else {

        assert(!(err->flags & PCIE_AER_ERR_TLP_PREFIX_PRESENT));

        memset(aer_cap + PCI_ERR_HEADER_LOG, 0, PCI_ERR_HEADER_LOG_SIZE);

    }



    if ((err->flags & PCIE_AER_ERR_TLP_PREFIX_PRESENT) &&

        (pci_get_long(dev->config + dev->exp.exp_cap + PCI_EXP_DEVCAP2) &

         PCI_EXP_DEVCAP2_EETLPP)) {

        for (i = 0; i < ARRAY_SIZE(err->prefix); ++i) {

            /* 7.10.12 tlp prefix log register */

            uint8_t *prefix_log =

                aer_cap + PCI_ERR_TLP_PREFIX_LOG + i * sizeof err->prefix[0];

            stl_be_p(prefix_log, err->prefix[i]);

        }

        errcap |= PCI_ERR_CAP_TLP;

    } else {

        memset(aer_cap + PCI_ERR_TLP_PREFIX_LOG, 0,

               PCI_ERR_TLP_PREFIX_LOG_SIZE);

    }

    pci_set_long(aer_cap + PCI_ERR_CAP, errcap);

}
