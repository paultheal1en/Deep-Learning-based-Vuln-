exynos4_boards_init_common(MachineState *machine,

                           Exynos4BoardType board_type)

{

    Exynos4BoardState *s = g_new(Exynos4BoardState, 1);

    MachineClass *mc = MACHINE_GET_CLASS(machine);



    if (smp_cpus != EXYNOS4210_NCPUS && !qtest_enabled()) {

        error_report("%s board supports only %d CPU cores, ignoring smp_cpus"

                     " value",

                     mc->name, EXYNOS4210_NCPUS);

    }



    exynos4_board_binfo.ram_size = exynos4_board_ram_size[board_type];

    exynos4_board_binfo.board_id = exynos4_board_id[board_type];

    exynos4_board_binfo.smp_bootreg_addr =

            exynos4_board_smp_bootreg_addr[board_type];

    exynos4_board_binfo.kernel_filename = machine->kernel_filename;

    exynos4_board_binfo.initrd_filename = machine->initrd_filename;

    exynos4_board_binfo.kernel_cmdline = machine->kernel_cmdline;

    exynos4_board_binfo.gic_cpu_if_addr =

            EXYNOS4210_SMP_PRIVATE_BASE_ADDR + 0x100;



    PRINT_DEBUG("\n ram_size: %luMiB [0x%08lx]\n"

            " kernel_filename: %s\n"

            " kernel_cmdline: %s\n"

            " initrd_filename: %s\n",

            exynos4_board_ram_size[board_type] / 1048576,

            exynos4_board_ram_size[board_type],

            machine->kernel_filename,

            machine->kernel_cmdline,

            machine->initrd_filename);



    exynos4_boards_init_ram(s, get_system_memory(),

                            exynos4_board_ram_size[board_type]);



    s->soc = exynos4210_init(get_system_memory());



    return s;

}
