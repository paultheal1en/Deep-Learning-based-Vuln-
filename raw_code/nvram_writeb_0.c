static void nvram_writeb (void *opaque, target_phys_addr_t addr, uint32_t value)

{

    M48t59State *NVRAM = opaque;



    m48t59_write(NVRAM, addr, value & 0xff);

}
