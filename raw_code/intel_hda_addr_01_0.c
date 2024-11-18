static target_phys_addr_t intel_hda_addr(uint32_t lbase, uint32_t ubase)

{

    target_phys_addr_t addr;



    addr = ((uint64_t)ubase << 32) | lbase;

    return addr;

}
