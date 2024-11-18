static void qpci_pc_config_writel(QPCIBus *bus, int devfn, uint8_t offset, uint32_t value)

{

    outl(0xcf8, (1 << 31) | (devfn << 8) | offset);

    outl(0xcfc, value);

}
