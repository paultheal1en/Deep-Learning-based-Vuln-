static void option_rom_setup_reset(target_phys_addr_t addr, unsigned size)

{

    RomResetData *rrd = qemu_malloc(sizeof *rrd);



    rrd->data = qemu_malloc(size);

    cpu_physical_memory_read(addr, rrd->data, size);

    rrd->addr = addr;

    rrd->size = size;

    qemu_register_reset(option_rom_reset, rrd);

}
