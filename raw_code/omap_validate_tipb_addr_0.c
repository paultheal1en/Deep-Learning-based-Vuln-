static int omap_validate_tipb_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return range_covers_byte(0xfffb0000, 0xffff0000 - 0xfffb0000, addr);

}
