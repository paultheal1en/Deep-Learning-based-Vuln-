static void bmdma_addr_write(void *opaque, target_phys_addr_t addr,

                             uint64_t data, unsigned width)

{

    BMDMAState *bm = opaque;

    int shift = addr * 8;

    uint32_t mask = (1ULL << (width * 8)) - 1;



#ifdef DEBUG_IDE

    printf("%s: 0x%08x\n", __func__, (unsigned)data);

#endif

    bm->addr &= ~(mask << shift);

    bm->addr |= ((data & mask) << shift) & ~3;

}
