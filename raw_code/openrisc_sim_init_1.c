static void openrisc_sim_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    OpenRISCCPU *cpu = NULL;

    MemoryRegion *ram;

    int n;



    if (!cpu_model) {

        cpu_model = "or1200";

    }



    for (n = 0; n < smp_cpus; n++) {

        cpu = OPENRISC_CPU(cpu_generic_init(TYPE_OPENRISC_CPU, cpu_model));

        if (cpu == NULL) {

            fprintf(stderr, "Unable to find CPU definition!\n");

            exit(1);

        }

        qemu_register_reset(main_cpu_reset, cpu);

        main_cpu_reset(cpu);

    }



    ram = g_malloc(sizeof(*ram));

    memory_region_init_ram(ram, NULL, "openrisc.ram", ram_size, &error_fatal);

    memory_region_add_subregion(get_system_memory(), 0, ram);



    cpu_openrisc_pic_init(cpu);

    cpu_openrisc_clock_init(cpu);



    serial_mm_init(get_system_memory(), 0x90000000, 0, cpu->env.irq[2],

                   115200, serial_hds[0], DEVICE_NATIVE_ENDIAN);



    if (nd_table[0].used) {

        openrisc_sim_net_init(get_system_memory(), 0x92000000,

                              0x92000400, cpu->env.irq[4], nd_table);

    }



    cpu_openrisc_load_kernel(ram_size, kernel_filename, cpu);

}
