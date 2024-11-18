sdhci_writefn(void *opaque, hwaddr off, uint64_t val, unsigned sz)

{

    SDHCIState *s = (SDHCIState *)opaque;



    SDHCI_GET_CLASS(s)->mem_write(s, off, val, sz);

}
