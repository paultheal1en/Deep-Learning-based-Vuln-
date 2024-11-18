static void spitz_common_init(MachineState *machine,

                              enum spitz_model_e model, int arm_id)

{

    PXA2xxState *mpu;

    DeviceState *scp0, *scp1 = NULL;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *rom = g_new(MemoryRegion, 1);

    const char *cpu_model = machine->cpu_model;



    if (!cpu_model)

        cpu_model = (model == terrier) ? "pxa270-c5" : "pxa270-c0";



    /* Setup CPU & memory */

    mpu = pxa270_init(address_space_mem, spitz_binfo.ram_size, cpu_model);



    sl_flash_register(mpu, (model == spitz) ? FLASH_128M : FLASH_1024M);



    memory_region_init_ram(rom, NULL, "spitz.rom", SPITZ_ROM, &error_abort);

    vmstate_register_ram_global(rom);

    memory_region_set_readonly(rom, true);

    memory_region_add_subregion(address_space_mem, 0, rom);



    /* Setup peripherals */

    spitz_keyboard_register(mpu);



    spitz_ssp_attach(mpu);



    scp0 = sysbus_create_simple("scoop", 0x10800000, NULL);

    if (model != akita) {

        scp1 = sysbus_create_simple("scoop", 0x08800040, NULL);

    }



    spitz_scoop_gpio_setup(mpu, scp0, scp1);



    spitz_gpio_setup(mpu, (model == akita) ? 1 : 2);



    spitz_i2c_setup(mpu);



    if (model == akita)

        spitz_akita_i2c_setup(mpu);



    if (model == terrier)

        /* A 6.0 GB microdrive is permanently sitting in CF slot 1.  */

        spitz_microdrive_attach(mpu, 1);

    else if (model != akita)

        /* A 4.0 GB microdrive is permanently sitting in CF slot 0.  */

        spitz_microdrive_attach(mpu, 0);



    spitz_binfo.kernel_filename = machine->kernel_filename;

    spitz_binfo.kernel_cmdline = machine->kernel_cmdline;

    spitz_binfo.initrd_filename = machine->initrd_filename;

    spitz_binfo.board_id = arm_id;

    arm_load_kernel(mpu->cpu, &spitz_binfo);

    sl_bootparam_write(SL_PXA_PARAM_BASE);

}
