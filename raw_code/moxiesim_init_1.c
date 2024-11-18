static void moxiesim_init(MachineState *machine)

{

    MoxieCPU *cpu = NULL;

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    CPUMoxieState *env;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *rom = g_new(MemoryRegion, 1);

    hwaddr ram_base = 0x200000;

    LoaderParams loader_params;



    /* Init CPUs. */

    if (cpu_model == NULL) {

        cpu_model = "MoxieLite-moxie-cpu";

    }

    cpu = MOXIE_CPU(cpu_generic_init(TYPE_MOXIE_CPU, cpu_model));

    if (!cpu) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    qemu_register_reset(main_cpu_reset, cpu);



    /* Allocate RAM. */

    memory_region_init_ram(ram, NULL, "moxiesim.ram", ram_size, &error_fatal);

    memory_region_add_subregion(address_space_mem, ram_base, ram);



    memory_region_init_ram(rom, NULL, "moxie.rom", 128 * 0x1000, &error_fatal);

    memory_region_add_subregion(get_system_memory(), 0x1000, rom);



    if (kernel_filename) {

        loader_params.ram_size = ram_size;

        loader_params.kernel_filename = kernel_filename;

        loader_params.kernel_cmdline = kernel_cmdline;

        loader_params.initrd_filename = initrd_filename;

        load_kernel(cpu, &loader_params);

    }



    /* A single 16450 sits at offset 0x3f8.  */

    if (serial_hds[0]) {

        serial_mm_init(address_space_mem, 0x3f8, 0, env->irq[4],

                       8000000/16, serial_hds[0], DEVICE_LITTLE_ENDIAN);

    }

}
