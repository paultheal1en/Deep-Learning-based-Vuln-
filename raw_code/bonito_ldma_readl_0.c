static uint64_t bonito_ldma_readl(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    uint32_t val;

    PCIBonitoState *s = opaque;



    val = ((uint32_t *)(&s->bonldma))[addr/sizeof(uint32_t)];



    return val;

}
