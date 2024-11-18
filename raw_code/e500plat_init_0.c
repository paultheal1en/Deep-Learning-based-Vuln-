static void e500plat_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *boot_device = args->boot_device;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    PPCE500Params params = {

        .ram_size = ram_size,

        .boot_device = boot_device,

        .kernel_filename = kernel_filename,

        .kernel_cmdline = kernel_cmdline,

        .initrd_filename = initrd_filename,

        .cpu_model = cpu_model,

        .pci_first_slot = 0x11,

        .pci_nr_slots = 2,

        .fixup_devtree = e500plat_fixup_devtree,

    };



    ppce500_init(&params);

}
