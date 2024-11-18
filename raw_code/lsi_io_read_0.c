static uint64_t lsi_io_read(void *opaque, target_phys_addr_t addr,

                            unsigned size)

{

    LSIState *s = opaque;

    return lsi_reg_readb(s, addr & 0xff);

}
