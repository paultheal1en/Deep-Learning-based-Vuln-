static void palmte_init(ram_addr_t ram_size, int vga_ram_size,

                const char *boot_device,

                const char *kernel_filename, const char *kernel_cmdline,

                const char *initrd_filename, const char *cpu_model)

{

    struct omap_mpu_state_s *cpu;

    int flash_size = 0x00800000;

    int sdram_size = palmte_binfo.ram_size;

    int io;

    static uint32_t cs0val = 0xffffffff;

    static uint32_t cs1val = 0x0000e1a0;

    static uint32_t cs2val = 0x0000e1a0;

    static uint32_t cs3val = 0xe1a0e1a0;

    ram_addr_t phys_flash;

    int rom_size, rom_loaded = 0;

    DisplayState *ds = get_displaystate();



    if (ram_size < flash_size + sdram_size + OMAP15XX_SRAM_SIZE) {

        fprintf(stderr, "This architecture uses %i bytes of memory\n",

                        flash_size + sdram_size + OMAP15XX_SRAM_SIZE);

        exit(1);

    }



    cpu = omap310_mpu_init(sdram_size, cpu_model);



    /* External Flash (EMIFS) */

    cpu_register_physical_memory(OMAP_CS0_BASE, flash_size,

                    (phys_flash = qemu_ram_alloc(flash_size)) | IO_MEM_ROM);



    io = cpu_register_io_memory(0, static_readfn, static_writefn, &cs0val);

    cpu_register_physical_memory(OMAP_CS0_BASE + flash_size,

                    OMAP_CS0_SIZE - flash_size, io);

    io = cpu_register_io_memory(0, static_readfn, static_writefn, &cs1val);

    cpu_register_physical_memory(OMAP_CS1_BASE, OMAP_CS1_SIZE, io);

    io = cpu_register_io_memory(0, static_readfn, static_writefn, &cs2val);

    cpu_register_physical_memory(OMAP_CS2_BASE, OMAP_CS2_SIZE, io);

    io = cpu_register_io_memory(0, static_readfn, static_writefn, &cs3val);

    cpu_register_physical_memory(OMAP_CS3_BASE, OMAP_CS3_SIZE, io);



    palmte_microwire_setup(cpu);



    qemu_add_kbd_event_handler(palmte_button_event, cpu);



    palmte_gpio_setup(cpu);



    /* Setup initial (reset) machine state */

    if (nb_option_roms) {

        rom_size = get_image_size(option_rom[0]);

        if (rom_size > flash_size) {

            fprintf(stderr, "%s: ROM image too big (%x > %x)\n",

                            __FUNCTION__, rom_size, flash_size);

            rom_size = 0;

        }

        if (rom_size > 0) {

            rom_size = load_image_targphys(option_rom[0], OMAP_CS0_BASE,

                                           flash_size);

            rom_loaded = 1;

            cpu->env->regs[15] = 0x00000000;

        }

        if (rom_size < 0) {

            fprintf(stderr, "%s: error loading '%s'\n",

                            __FUNCTION__, option_rom[0]);

        }

    }



    if (!rom_loaded && !kernel_filename) {

        fprintf(stderr, "Kernel or ROM image must be specified\n");

        exit(1);

    }



    /* Load the kernel.  */

    if (kernel_filename) {

        /* Start at bootloader.  */

        cpu->env->regs[15] = palmte_binfo.loader_start;



        palmte_binfo.kernel_filename = kernel_filename;

        palmte_binfo.kernel_cmdline = kernel_cmdline;

        palmte_binfo.initrd_filename = initrd_filename;

        arm_load_kernel(cpu->env, &palmte_binfo);

    }



    /* FIXME: We shouldn't really be doing this here.  The LCD controller

       will set the size once configured, so this just sets an initial

       size until the guest activates the display.  */

    ds->surface = qemu_resize_displaysurface(ds, 320, 320);

    dpy_resize(ds);

}
