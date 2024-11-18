void stq_tce(VIOsPAPRDevice *dev, uint64_t taddr, uint64_t val)

{

    val = tswap64(val);

    spapr_tce_dma_write(dev, taddr, &val, sizeof(val));

}
