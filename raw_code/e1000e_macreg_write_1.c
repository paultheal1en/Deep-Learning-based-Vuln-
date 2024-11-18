static void e1000e_macreg_write(e1000e_device *d, uint32_t reg, uint32_t val)

{

    qpci_io_writel(d->pci_dev, d->mac_regs + reg, val);

}
