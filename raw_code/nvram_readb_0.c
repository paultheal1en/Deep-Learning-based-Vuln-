static uint32_t nvram_readb (void *opaque, target_phys_addr_t addr)

{

    M48t59State *NVRAM = opaque;

    uint32_t retval;



    retval = m48t59_read(NVRAM, addr);

    return retval;

}
