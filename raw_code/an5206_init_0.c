static void an5206_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    M68kCPU *cpu;

    CPUM68KState *env;

    int kernel_size;

    uint64_t elf_entry;

    hwaddr entry;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *sram = g_new(MemoryRegion, 1);



    if (!cpu_model) {

        cpu_model = "m5206";

    }

    cpu = M68K_CPU(cpu_generic_init(TYPE_M68K_CPU, cpu_model));

    env = &cpu->env;



    /* Initialize CPU registers.  */

    env->vbr = 0;

    /* TODO: allow changing MBAR and RAMBAR.  */

    env->mbar = AN5206_MBAR_ADDR | 1;

    env->rambar0 = AN5206_RAMBAR_ADDR | 1;



    /* DRAM at address zero */

    memory_region_allocate_system_memory(ram, NULL, "an5206.ram", ram_size);

    memory_region_add_subregion(address_space_mem, 0, ram);



    /* Internal SRAM.  */

    memory_region_init_ram(sram, NULL, "an5206.sram", 512, &error_fatal);

    memory_region_add_subregion(address_space_mem, AN5206_RAMBAR_ADDR, sram);



    mcf5206_init(address_space_mem, AN5206_MBAR_ADDR, cpu);



    /* Load kernel.  */

    if (!kernel_filename) {

        if (qtest_enabled()) {

            return;

        }

        fprintf(stderr, "Kernel image must be specified\n");

        exit(1);

    }



    kernel_size = load_elf(kernel_filename, NULL, NULL, &elf_entry,

                           NULL, NULL, 1, EM_68K, 0, 0);

    entry = elf_entry;

    if (kernel_size < 0) {

        kernel_size = load_uimage(kernel_filename, &entry, NULL, NULL,

                                  NULL, NULL);

    }

    if (kernel_size < 0) {

        kernel_size = load_image_targphys(kernel_filename, KERNEL_LOAD_ADDR,

                                          ram_size - KERNEL_LOAD_ADDR);

        entry = KERNEL_LOAD_ADDR;

    }

    if (kernel_size < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n", kernel_filename);

        exit(1);

    }



    env->pc = entry;

}
