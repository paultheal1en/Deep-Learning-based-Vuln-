mips_mipssim_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    char *filename;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *isa = g_new(MemoryRegion, 1);

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *bios = g_new(MemoryRegion, 1);

    MIPSCPU *cpu;

    CPUMIPSState *env;

    ResetData *reset_info;

    int bios_size;



    /* Init CPUs. */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "5Kf";

#else

        cpu_model = "24Kf";

#endif

    }

    cpu = cpu_mips_init(cpu_model);

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    reset_info = g_malloc0(sizeof(ResetData));

    reset_info->cpu = cpu;

    reset_info->vector = env->active_tc.PC;

    qemu_register_reset(main_cpu_reset, reset_info);



    /* Allocate RAM. */

    memory_region_allocate_system_memory(ram, NULL, "mips_mipssim.ram",

                                         ram_size);

    memory_region_init_ram(bios, NULL, "mips_mipssim.bios", BIOS_SIZE,

                           &error_abort);

    vmstate_register_ram_global(bios);

    memory_region_set_readonly(bios, true);



    memory_region_add_subregion(address_space_mem, 0, ram);



    /* Map the BIOS / boot exception handler. */

    memory_region_add_subregion(address_space_mem, 0x1fc00000LL, bios);

    /* Load a BIOS / boot exception handler image. */

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        bios_size = load_image_targphys(filename, 0x1fc00000LL, BIOS_SIZE);

        g_free(filename);

    } else {

        bios_size = -1;

    }

    if ((bios_size < 0 || bios_size > BIOS_SIZE) &&

        !kernel_filename && !qtest_enabled()) {

        /* Bail out if we have neither a kernel image nor boot vector code. */

        error_report("Could not load MIPS bios '%s', and no "

                     "-kernel argument was specified", bios_name);

        exit(1);

    } else {

        /* We have a boot vector start address. */

        env->active_tc.PC = (target_long)(int32_t)0xbfc00000;

    }



    if (kernel_filename) {

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        reset_info->vector = load_kernel();

    }



    /* Init CPU internal devices. */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* Register 64 KB of ISA IO space at 0x1fd00000. */

    memory_region_init_alias(isa, NULL, "isa_mmio",

                             get_system_io(), 0, 0x00010000);

    memory_region_add_subregion(get_system_memory(), 0x1fd00000, isa);



    /* A single 16450 sits at offset 0x3f8. It is attached to

       MIPS CPU INT2, which is interrupt 4. */

    if (serial_hds[0])

        serial_init(0x3f8, env->irq[4], 115200, serial_hds[0],

                    get_system_io());



    if (nd_table[0].used)

        /* MIPSnet uses the MIPS CPU INT0, which is interrupt 2. */

        mipsnet_init(0x4200, env->irq[2], &nd_table[0]);

}
