static void tricore_testboard_init(MachineState *machine, int board_id)

{

    TriCoreCPU *cpu;

    CPUTriCoreState *env;



    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ext_cram = g_new(MemoryRegion, 1);

    MemoryRegion *ext_dram = g_new(MemoryRegion, 1);

    MemoryRegion *int_cram = g_new(MemoryRegion, 1);

    MemoryRegion *int_dram = g_new(MemoryRegion, 1);

    MemoryRegion *pcp_data = g_new(MemoryRegion, 1);

    MemoryRegion *pcp_text = g_new(MemoryRegion, 1);



    if (!machine->cpu_model) {

        machine->cpu_model = "tc1796";

    }

    cpu = cpu_tricore_init(machine->cpu_model);

    if (!cpu) {

        error_report("Unable to find CPU definition");

        exit(1);

    }

    env = &cpu->env;

    memory_region_init_ram(ext_cram, NULL, "powerlink_ext_c.ram", 2*1024*1024, &error_abort);

    vmstate_register_ram_global(ext_cram);

    memory_region_init_ram(ext_dram, NULL, "powerlink_ext_d.ram", 4*1024*1024, &error_abort);

    vmstate_register_ram_global(ext_dram);

    memory_region_init_ram(int_cram, NULL, "powerlink_int_c.ram", 48*1024, &error_abort);

    vmstate_register_ram_global(int_cram);

    memory_region_init_ram(int_dram, NULL, "powerlink_int_d.ram", 48*1024, &error_abort);

    vmstate_register_ram_global(int_dram);

    memory_region_init_ram(pcp_data, NULL, "powerlink_pcp_data.ram", 16*1024, &error_abort);

    vmstate_register_ram_global(pcp_data);

    memory_region_init_ram(pcp_text, NULL, "powerlink_pcp_text.ram", 32*1024, &error_abort);

    vmstate_register_ram_global(pcp_text);



    memory_region_add_subregion(sysmem, 0x80000000, ext_cram);

    memory_region_add_subregion(sysmem, 0xa1000000, ext_dram);

    memory_region_add_subregion(sysmem, 0xd4000000, int_cram);

    memory_region_add_subregion(sysmem, 0xd0000000, int_dram);

    memory_region_add_subregion(sysmem, 0xf0050000, pcp_data);

    memory_region_add_subregion(sysmem, 0xf0060000, pcp_text);



    tricoretb_binfo.ram_size = machine->ram_size;

    tricoretb_binfo.kernel_filename = machine->kernel_filename;



    if (machine->kernel_filename) {

        tricore_load_kernel(env);

    }

}
