void *rom_ptr(target_phys_addr_t addr)

{

    Rom *rom;



    rom = find_rom(addr);

    if (!rom || !rom->data)

        return NULL;

    return rom->data + (addr - rom->addr);

}
