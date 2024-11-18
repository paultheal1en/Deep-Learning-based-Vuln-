void qpci_msix_disable(QPCIDevice *dev)

{

    uint8_t addr;

    uint16_t val;



    g_assert(dev->msix_enabled);

    addr = qpci_find_capability(dev, PCI_CAP_ID_MSIX);

    g_assert_cmphex(addr, !=, 0);

    val = qpci_config_readw(dev, addr + PCI_MSIX_FLAGS);

    qpci_config_writew(dev, addr + PCI_MSIX_FLAGS,

                                                val & ~PCI_MSIX_FLAGS_ENABLE);



    qpci_iounmap(dev, dev->msix_table);

    qpci_iounmap(dev, dev->msix_pba);

    dev->msix_enabled = 0;

    dev->msix_table = NULL;

    dev->msix_pba = NULL;

}
