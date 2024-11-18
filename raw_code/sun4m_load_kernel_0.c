static void sun4m_load_kernel(long vram_size, int ram_size, int boot_device,

                              const char *kernel_filename,

                              const char *kernel_cmdline,

                              const char *initrd_filename,

                              int machine_id)

{

    int ret, linux_boot;

    char buf[1024];

    unsigned int i;

    long prom_offset, initrd_size, kernel_size;



    linux_boot = (kernel_filename != NULL);



    prom_offset = ram_size + vram_size;

    cpu_register_physical_memory(PROM_ADDR, 

                                 (PROM_SIZE_MAX + TARGET_PAGE_SIZE - 1) & TARGET_PAGE_MASK, 

                                 prom_offset | IO_MEM_ROM);



    snprintf(buf, sizeof(buf), "%s/%s", bios_dir, PROM_FILENAME);

    ret = load_elf(buf, 0, NULL, NULL, NULL);

    if (ret < 0) {

	fprintf(stderr, "qemu: could not load prom '%s'\n", 

		buf);

	exit(1);

    }



    kernel_size = 0;

    if (linux_boot) {

        kernel_size = load_elf(kernel_filename, -0xf0000000, NULL, NULL, NULL);

        if (kernel_size < 0)

	    kernel_size = load_aout(kernel_filename, phys_ram_base + KERNEL_LOAD_ADDR);

	if (kernel_size < 0)

	    kernel_size = load_image(kernel_filename, phys_ram_base + KERNEL_LOAD_ADDR);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n", 

                    kernel_filename);

	    exit(1);

        }



        /* load initrd */

        initrd_size = 0;

        if (initrd_filename) {

            initrd_size = load_image(initrd_filename, phys_ram_base + INITRD_LOAD_ADDR);

            if (initrd_size < 0) {

                fprintf(stderr, "qemu: could not load initial ram disk '%s'\n", 

                        initrd_filename);

                exit(1);

            }

        }

        if (initrd_size > 0) {

	    for (i = 0; i < 64 * TARGET_PAGE_SIZE; i += TARGET_PAGE_SIZE) {

		if (ldl_raw(phys_ram_base + KERNEL_LOAD_ADDR + i)

		    == 0x48647253) { // HdrS

		    stl_raw(phys_ram_base + KERNEL_LOAD_ADDR + i + 16, INITRD_LOAD_ADDR);

		    stl_raw(phys_ram_base + KERNEL_LOAD_ADDR + i + 20, initrd_size);

		    break;

		}

	    }

        }

    }

    nvram_init(nvram, (uint8_t *)&nd_table[0].macaddr, kernel_cmdline,

               boot_device, ram_size, kernel_size, graphic_width,

               graphic_height, graphic_depth, machine_id);

}
