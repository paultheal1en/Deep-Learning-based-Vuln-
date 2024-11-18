static void nvram_writew (void *opaque, target_phys_addr_t addr, uint32_t value)

{

    M48t59State *NVRAM = opaque;



    m48t59_write(NVRAM, addr, (value >> 8) & 0xff);

    m48t59_write(NVRAM, addr + 1, value & 0xff);

}
