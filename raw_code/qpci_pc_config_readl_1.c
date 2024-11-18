static uint32_t qpci_pc_config_readl(QPCIBus *bus, int devfn, uint8_t offset)

{

    outl(0xcf8, (1 << 31) | (devfn << 8) | offset);

    return inl(0xcfc);

}
