static void lm32_uclinux_init(QEMUMachineInitArgs *args)
{
    const char *cpu_model = args->cpu_model;
    const char *kernel_filename = args->kernel_filename;
    const char *kernel_cmdline = args->kernel_cmdline;
    const char *initrd_filename = args->initrd_filename;
    LM32CPU *cpu;
    CPULM32State *env;
    DriveInfo *dinfo;
    MemoryRegion *address_space_mem =  get_system_memory();
    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);
    qemu_irq *cpu_irq, irq[32];
    HWSetup *hw;
    ResetInfo *reset_info;
    int i;
    /* memory map */
    hwaddr flash_base   = 0x04000000;
    size_t flash_sector_size        = 256 * 1024;
    size_t flash_size               = 32 * 1024 * 1024;
    hwaddr ram_base     = 0x08000000;
    size_t ram_size                 = 64 * 1024 * 1024;
    hwaddr uart0_base   = 0x80000000;
    hwaddr timer0_base  = 0x80002000;
    hwaddr timer1_base  = 0x80010000;
    hwaddr timer2_base  = 0x80012000;
    int uart0_irq                   = 0;
    int timer0_irq                  = 1;
    int timer1_irq                  = 20;
    int timer2_irq                  = 21;
    hwaddr hwsetup_base = 0x0bffe000;
    hwaddr cmdline_base = 0x0bfff000;
    hwaddr initrd_base  = 0x08400000;
    size_t initrd_max               = 0x01000000;
    reset_info = g_malloc0(sizeof(ResetInfo));
    if (cpu_model == NULL) {
        cpu_model = "lm32-full";
    cpu = cpu_lm32_init(cpu_model);
    env = &cpu->env;
    reset_info->cpu = cpu;
    reset_info->flash_base = flash_base;
    memory_region_init_ram(phys_ram, NULL, "lm32_uclinux.sdram", ram_size);
    vmstate_register_ram_global(phys_ram);
    memory_region_add_subregion(address_space_mem, ram_base, phys_ram);
    dinfo = drive_get(IF_PFLASH, 0, 0);
    /* Spansion S29NS128P */
    pflash_cfi02_register(flash_base, NULL, "lm32_uclinux.flash", flash_size,
                          dinfo ? dinfo->bdrv : NULL, flash_sector_size,
                          flash_size / flash_sector_size, 1, 2,
                          0x01, 0x7e, 0x43, 0x00, 0x555, 0x2aa, 1);
    /* create irq lines */
    cpu_irq = qemu_allocate_irqs(cpu_irq_handler, env, 1);
    env->pic_state = lm32_pic_init(*cpu_irq);
    for (i = 0; i < 32; i++) {
        irq[i] = qdev_get_gpio_in(env->pic_state, i);
    sysbus_create_simple("lm32-uart", uart0_base, irq[uart0_irq]);
    sysbus_create_simple("lm32-timer", timer0_base, irq[timer0_irq]);
    sysbus_create_simple("lm32-timer", timer1_base, irq[timer1_irq]);
    sysbus_create_simple("lm32-timer", timer2_base, irq[timer2_irq]);
    /* make sure juart isn't the first chardev */
    env->juart_state = lm32_juart_init();
    reset_info->bootstrap_pc = flash_base;
    if (kernel_filename) {
        uint64_t entry;
        int kernel_size;
        kernel_size = load_elf(kernel_filename, NULL, NULL, &entry, NULL, NULL,
                               1, ELF_MACHINE, 0);
        reset_info->bootstrap_pc = entry;
        if (kernel_size < 0) {
            kernel_size = load_image_targphys(kernel_filename, ram_base,
                                              ram_size);
            reset_info->bootstrap_pc = ram_base;
        if (kernel_size < 0) {
            fprintf(stderr, "qemu: could not load kernel '%s'\n",
                    kernel_filename);
    /* generate a rom with the hardware description */
    hw = hwsetup_init();
    hwsetup_add_cpu(hw, "LM32", 75000000);
    hwsetup_add_flash(hw, "flash", flash_base, flash_size);
    hwsetup_add_ddr_sdram(hw, "ddr_sdram", ram_base, ram_size);
    hwsetup_add_timer(hw, "timer0", timer0_base, timer0_irq);
    hwsetup_add_timer(hw, "timer1_dev_only", timer1_base, timer1_irq);
    hwsetup_add_timer(hw, "timer2_dev_only", timer2_base, timer2_irq);
    hwsetup_add_uart(hw, "uart", uart0_base, uart0_irq);
    hwsetup_add_trailer(hw);
    hwsetup_create_rom(hw, hwsetup_base);
    hwsetup_free(hw);
    reset_info->hwsetup_base = hwsetup_base;
    if (kernel_cmdline && strlen(kernel_cmdline)) {
        pstrcpy_targphys("cmdline", cmdline_base, TARGET_PAGE_SIZE,
                kernel_cmdline);
        reset_info->cmdline_base = cmdline_base;
    if (initrd_filename) {
        size_t initrd_size;
        initrd_size = load_image_targphys(initrd_filename, initrd_base,
                initrd_max);
        reset_info->initrd_base = initrd_base;
        reset_info->initrd_size = initrd_size;
    qemu_register_reset(main_cpu_reset, reset_info);