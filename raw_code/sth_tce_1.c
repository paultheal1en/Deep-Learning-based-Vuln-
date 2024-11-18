void sth_tce(VIOsPAPRDevice *dev, uint64_t taddr, uint16_t val)

{

    val = tswap16(val);

    spapr_tce_dma_write(dev, taddr, &val, sizeof(val));

}
