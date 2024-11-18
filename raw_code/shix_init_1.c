static void shix_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    int ret;

    SuperHCPU *cpu;

    struct SH7750State *s;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *rom = g_new(MemoryRegion, 1);

    MemoryRegion *sdram = g_new(MemoryRegion, 2);

    

    if (!cpu_model)

        cpu_model = "any";



    cpu = SUPERH_CPU(cpu_generic_init(TYPE_SUPERH_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



    /* Allocate memory space */

    memory_region_init_ram(rom, NULL, "shix.rom", 0x4000, &error_fatal);

    memory_region_set_readonly(rom, true);

    memory_region_add_subregion(sysmem, 0x00000000, rom);

    memory_region_init_ram(&sdram[0], NULL, "shix.sdram1", 0x01000000,

                           &error_fatal);

    memory_region_add_subregion(sysmem, 0x08000000, &sdram[0]);

    memory_region_init_ram(&sdram[1], NULL, "shix.sdram2", 0x01000000,

                           &error_fatal);

    memory_region_add_subregion(sysmem, 0x0c000000, &sdram[1]);



    /* Load BIOS in 0 (and access it through P2, 0xA0000000) */

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    ret = load_image_targphys(bios_name, 0, 0x4000);

    if (ret < 0 && !qtest_enabled()) {

        error_report("Could not load SHIX bios '%s'", bios_name);

        exit(1);

    }



    /* Register peripherals */

    s = sh7750_init(cpu, sysmem);

    /* XXXXX Check success */

    tc58128_init(s, "shix_linux_nand.bin", NULL);

}
