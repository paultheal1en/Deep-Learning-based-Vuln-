int omap_validate_tipb_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= 0xfffb0000 && addr < 0xffff0000;

}
