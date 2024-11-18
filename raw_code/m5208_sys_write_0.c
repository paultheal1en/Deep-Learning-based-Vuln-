static void m5208_sys_write(void *opaque, target_phys_addr_t addr,

                            uint64_t value, unsigned size)

{

    hw_error("m5208_sys_write: Bad offset 0x%x\n", (int)addr);

}
