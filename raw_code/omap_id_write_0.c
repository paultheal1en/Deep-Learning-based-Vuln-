static void omap_id_write(void *opaque, target_phys_addr_t addr,

                          uint64_t value, unsigned size)

{

    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    OMAP_BAD_REG(addr);

}
