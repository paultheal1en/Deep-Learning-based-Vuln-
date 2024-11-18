static void option_rom_reset(void *_rrd)

{

    RomResetData *rrd = _rrd;



    cpu_physical_memory_write_rom(rrd->addr, rrd->data, rrd->size);

}
