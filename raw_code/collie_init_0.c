static void collie_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    StrongARMState *s;

    DriveInfo *dinfo;

    MemoryRegion *sysmem = get_system_memory();



    if (!cpu_model) {

        cpu_model = "sa1110";

    }



    s = sa1110_init(sysmem, collie_binfo.ram_size, cpu_model);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    pflash_cfi01_register(SA_CS0, NULL, "collie.fl1", 0x02000000,

                    dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                    (64 * 1024), 512, 4, 0x00, 0x00, 0x00, 0x00, 0);



    dinfo = drive_get(IF_PFLASH, 0, 1);

    pflash_cfi01_register(SA_CS1, NULL, "collie.fl2", 0x02000000,

                    dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                    (64 * 1024), 512, 4, 0x00, 0x00, 0x00, 0x00, 0);



    sysbus_create_simple("scoop", 0x40800000, NULL);



    collie_binfo.kernel_filename = kernel_filename;

    collie_binfo.kernel_cmdline = kernel_cmdline;

    collie_binfo.initrd_filename = initrd_filename;

    collie_binfo.board_id = 0x208;

    arm_load_kernel(s->cpu, &collie_binfo);

}
