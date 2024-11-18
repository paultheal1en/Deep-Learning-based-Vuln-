void qusb_pci_init_one(QPCIBus *pcibus, struct qhc *hc, uint32_t devfn, int bar)

{

    hc->dev = qpci_device_find(pcibus, devfn);

    g_assert(hc->dev != NULL);

    qpci_device_enable(hc->dev);

    hc->base = qpci_iomap(hc->dev, bar, NULL);

    g_assert(hc->base != NULL);

}
