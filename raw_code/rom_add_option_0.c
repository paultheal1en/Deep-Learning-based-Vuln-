int rom_add_option(const char *file)

{

    if (!rom_enable_driver_roms)

        return 0;

    return rom_add_file(file, "genroms", file, 0);

}
