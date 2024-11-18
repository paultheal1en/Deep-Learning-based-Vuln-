static void spitz_common_init(ram_addr_t ram_size, int vga_ram_size,

                const char *kernel_filename,

                const char *kernel_cmdline, const char *initrd_filename,

                const char *cpu_model, enum spitz_model_e model, int arm_id)

{

    struct pxa2xx_state_s *cpu;

    struct scoop_info_s *scp0, *scp1 = NULL;



    if (!cpu_model)

        cpu_model = (model == terrier) ? "pxa270-c5" : "pxa270-c0";



    /* Setup CPU & memory */

    if (ram_size < SPITZ_RAM + SPITZ_ROM + PXA2XX_INTERNAL_SIZE) {

        fprintf(stderr, "This platform requires %i bytes of memory\n",

                        SPITZ_RAM + SPITZ_ROM + PXA2XX_INTERNAL_SIZE);

        exit(1);

    }

    cpu = pxa270_init(spitz_binfo.ram_size, cpu_model);



    sl_flash_register(cpu, (model == spitz) ? FLASH_128M : FLASH_1024M);



    cpu_register_physical_memory(0, SPITZ_ROM,

                    qemu_ram_alloc(SPITZ_ROM) | IO_MEM_ROM);



    /* Setup peripherals */

    spitz_keyboard_register(cpu);



    spitz_ssp_attach(cpu);



    scp0 = scoop_init(cpu, 0, 0x10800000);

    if (model != akita) {

	    scp1 = scoop_init(cpu, 1, 0x08800040);

    }



    spitz_scoop_gpio_setup(cpu, scp0, scp1);



    spitz_gpio_setup(cpu, (model == akita) ? 1 : 2);



    spitz_i2c_setup(cpu);



    if (model == akita)

        spitz_akita_i2c_setup(cpu);



    if (model == terrier)

        /* A 6.0 GB microdrive is permanently sitting in CF slot 1.  */

        spitz_microdrive_attach(cpu, 1);

    else if (model != akita)

        /* A 4.0 GB microdrive is permanently sitting in CF slot 0.  */

        spitz_microdrive_attach(cpu, 0);



    /* Setup initial (reset) machine state */

    cpu->env->regs[15] = spitz_binfo.loader_start;



    spitz_binfo.kernel_filename = kernel_filename;

    spitz_binfo.kernel_cmdline = kernel_cmdline;

    spitz_binfo.initrd_filename = initrd_filename;

    spitz_binfo.board_id = arm_id;

    arm_load_kernel(cpu->env, &spitz_binfo);

    sl_bootparam_write(SL_PXA_PARAM_BASE);

}
