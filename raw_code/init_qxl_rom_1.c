static void init_qxl_rom(PCIQXLDevice *d)

{

    QXLRom *rom = memory_region_get_ram_ptr(&d->rom_bar);

    QXLModes *modes = (QXLModes *)(rom + 1);

    uint32_t ram_header_size;

    uint32_t surface0_area_size;

    uint32_t num_pages;

    uint32_t fb, maxfb = 0;

    int i;



    memset(rom, 0, d->rom_size);



    rom->magic         = cpu_to_le32(QXL_ROM_MAGIC);

    rom->id            = cpu_to_le32(d->id);

    rom->log_level     = cpu_to_le32(d->guestdebug);

    rom->modes_offset  = cpu_to_le32(sizeof(QXLRom));



    rom->slot_gen_bits = MEMSLOT_GENERATION_BITS;

    rom->slot_id_bits  = MEMSLOT_SLOT_BITS;

    rom->slots_start   = 1;

    rom->slots_end     = NUM_MEMSLOTS - 1;

    rom->n_surfaces    = cpu_to_le32(NUM_SURFACES);



    modes->n_modes     = cpu_to_le32(ARRAY_SIZE(qxl_modes));

    for (i = 0; i < modes->n_modes; i++) {

        fb = qxl_modes[i].y_res * qxl_modes[i].stride;

        if (maxfb < fb) {

            maxfb = fb;

        }

        modes->modes[i].id          = cpu_to_le32(i);

        modes->modes[i].x_res       = cpu_to_le32(qxl_modes[i].x_res);

        modes->modes[i].y_res       = cpu_to_le32(qxl_modes[i].y_res);

        modes->modes[i].bits        = cpu_to_le32(qxl_modes[i].bits);

        modes->modes[i].stride      = cpu_to_le32(qxl_modes[i].stride);

        modes->modes[i].x_mili      = cpu_to_le32(qxl_modes[i].x_mili);

        modes->modes[i].y_mili      = cpu_to_le32(qxl_modes[i].y_mili);

        modes->modes[i].orientation = cpu_to_le32(qxl_modes[i].orientation);

    }

    if (maxfb < VGA_RAM_SIZE && d->id == 0)

        maxfb = VGA_RAM_SIZE;



    ram_header_size    = ALIGN(sizeof(QXLRam), 4096);

    surface0_area_size = ALIGN(maxfb, 4096);

    num_pages          = d->vga.vram_size;

    num_pages         -= ram_header_size;

    num_pages         -= surface0_area_size;

    num_pages          = num_pages / TARGET_PAGE_SIZE;



    rom->draw_area_offset   = cpu_to_le32(0);

    rom->surface0_area_size = cpu_to_le32(surface0_area_size);

    rom->pages_offset       = cpu_to_le32(surface0_area_size);

    rom->num_pages          = cpu_to_le32(num_pages);

    rom->ram_header_offset  = cpu_to_le32(d->vga.vram_size - ram_header_size);



    d->shadow_rom = *rom;

    d->rom        = rom;

    d->modes      = modes;

}
