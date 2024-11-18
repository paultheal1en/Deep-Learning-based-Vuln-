static uint64_t dchip_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    /* Skip this.  It's all related to DRAM timing and setup.  */

    return 0;

}
