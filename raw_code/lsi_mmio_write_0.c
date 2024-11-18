static void lsi_mmio_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned size)

{

    LSIState *s = opaque;



    lsi_reg_writeb(s, addr & 0xff, val);

}
