static void gt64120_write_config(PCIDevice *d, uint32_t address, uint32_t val,

                                 int len)

{

#ifdef TARGET_WORDS_BIGENDIAN

    val = bswap32(val);

#endif

    pci_default_write_config(d, address, val, len);

}
