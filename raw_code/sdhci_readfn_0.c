sdhci_readfn(void *opaque, hwaddr offset, unsigned size)

{

    SDHCIState *s = (SDHCIState *)opaque;



    return SDHCI_GET_CLASS(s)->mem_read(s, offset, size);

}
