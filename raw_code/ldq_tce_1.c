uint64_t ldq_tce(VIOsPAPRDevice *dev, uint64_t taddr)

{

    uint64_t val;



    spapr_tce_dma_read(dev, taddr, &val, sizeof(val));

    return tswap64(val);

}
