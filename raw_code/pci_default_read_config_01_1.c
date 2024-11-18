uint32_t pci_default_read_config(PCIDevice *d,

                                 uint32_t address, int len)

{

    uint32_t val = 0;

    assert(len == 1 || len == 2 || len == 4);

    len = MIN(len, pci_config_size(d) - address);

    memcpy(&val, d->config + address, len);

    return le32_to_cpu(val);

}
