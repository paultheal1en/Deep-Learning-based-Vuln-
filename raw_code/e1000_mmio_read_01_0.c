e1000_mmio_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    E1000State *s = opaque;

    unsigned int index = (addr & 0x1ffff) >> 2;



    if (index < NREADOPS && macreg_readops[index])

    {

        return macreg_readops[index](s, index);

    }

    DBGOUT(UNKNOWN, "MMIO unknown read addr=0x%08x\n", index<<2);

    return 0;

}
