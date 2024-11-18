int rom_add_vga(const char *file)

{

    if (!rom_enable_driver_roms)

        return 0;

    return rom_add_file(file, "vgaroms", file, 0);

}
