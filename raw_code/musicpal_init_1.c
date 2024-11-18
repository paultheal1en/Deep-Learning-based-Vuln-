static void musicpal_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    ARMCPU *cpu;

    qemu_irq pic[32];

    DeviceState *dev;

    DeviceState *i2c_dev;

    DeviceState *lcd_dev;

    DeviceState *key_dev;

    DeviceState *wm8750_dev;

    SysBusDevice *s;

    I2CBus *i2c;

    int i;

    unsigned long flash_size;

    DriveInfo *dinfo;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *sram = g_new(MemoryRegion, 1);



    if (!cpu_model) {

        cpu_model = "arm926";

    }

    cpu = ARM_CPU(cpu_generic_init(TYPE_ARM_CPU, cpu_model));

    if (!cpu) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



    /* For now we use a fixed - the original - RAM size */

    memory_region_allocate_system_memory(ram, NULL, "musicpal.ram",

                                         MP_RAM_DEFAULT_SIZE);

    memory_region_add_subregion(address_space_mem, 0, ram);



    memory_region_init_ram(sram, NULL, "musicpal.sram", MP_SRAM_SIZE,

                           &error_fatal);

    memory_region_add_subregion(address_space_mem, MP_SRAM_BASE, sram);



    dev = sysbus_create_simple(TYPE_MV88W8618_PIC, MP_PIC_BASE,

                               qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_IRQ));

    for (i = 0; i < 32; i++) {

        pic[i] = qdev_get_gpio_in(dev, i);

    }

    sysbus_create_varargs(TYPE_MV88W8618_PIT, MP_PIT_BASE, pic[MP_TIMER1_IRQ],

                          pic[MP_TIMER2_IRQ], pic[MP_TIMER3_IRQ],

                          pic[MP_TIMER4_IRQ], NULL);



    if (serial_hds[0]) {

        serial_mm_init(address_space_mem, MP_UART1_BASE, 2, pic[MP_UART1_IRQ],

                       1825000, serial_hds[0], DEVICE_NATIVE_ENDIAN);

    }

    if (serial_hds[1]) {

        serial_mm_init(address_space_mem, MP_UART2_BASE, 2, pic[MP_UART2_IRQ],

                       1825000, serial_hds[1], DEVICE_NATIVE_ENDIAN);

    }



    /* Register flash */

    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (dinfo) {

        BlockBackend *blk = blk_by_legacy_dinfo(dinfo);



        flash_size = blk_getlength(blk);

        if (flash_size != 8*1024*1024 && flash_size != 16*1024*1024 &&

            flash_size != 32*1024*1024) {

            fprintf(stderr, "Invalid flash image size\n");

            exit(1);

        }



        /*

         * The original U-Boot accesses the flash at 0xFE000000 instead of

         * 0xFF800000 (if there is 8 MB flash). So remap flash access if the

         * image is smaller than 32 MB.

         */

#ifdef TARGET_WORDS_BIGENDIAN

        pflash_cfi02_register(0x100000000ULL-MP_FLASH_SIZE_MAX, NULL,

                              "musicpal.flash", flash_size,

                              blk, 0x10000, (flash_size + 0xffff) >> 16,

                              MP_FLASH_SIZE_MAX / flash_size,

                              2, 0x00BF, 0x236D, 0x0000, 0x0000,

                              0x5555, 0x2AAA, 1);

#else

        pflash_cfi02_register(0x100000000ULL-MP_FLASH_SIZE_MAX, NULL,

                              "musicpal.flash", flash_size,

                              blk, 0x10000, (flash_size + 0xffff) >> 16,

                              MP_FLASH_SIZE_MAX / flash_size,

                              2, 0x00BF, 0x236D, 0x0000, 0x0000,

                              0x5555, 0x2AAA, 0);

#endif



    }

    sysbus_create_simple(TYPE_MV88W8618_FLASHCFG, MP_FLASHCFG_BASE, NULL);



    qemu_check_nic_model(&nd_table[0], "mv88w8618");

    dev = qdev_create(NULL, TYPE_MV88W8618_ETH);

    qdev_set_nic_properties(dev, &nd_table[0]);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, MP_ETH_BASE);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, pic[MP_ETH_IRQ]);



    sysbus_create_simple("mv88w8618_wlan", MP_WLAN_BASE, NULL);



    sysbus_create_simple(TYPE_MUSICPAL_MISC, MP_MISC_BASE, NULL);



    dev = sysbus_create_simple(TYPE_MUSICPAL_GPIO, MP_GPIO_BASE,

                               pic[MP_GPIO_IRQ]);

    i2c_dev = sysbus_create_simple("gpio_i2c", -1, NULL);

    i2c = (I2CBus *)qdev_get_child_bus(i2c_dev, "i2c");



    lcd_dev = sysbus_create_simple(TYPE_MUSICPAL_LCD, MP_LCD_BASE, NULL);

    key_dev = sysbus_create_simple(TYPE_MUSICPAL_KEY, -1, NULL);



    /* I2C read data */

    qdev_connect_gpio_out(i2c_dev, 0,

                          qdev_get_gpio_in(dev, MP_GPIO_I2C_DATA_BIT));

    /* I2C data */

    qdev_connect_gpio_out(dev, 3, qdev_get_gpio_in(i2c_dev, 0));

    /* I2C clock */

    qdev_connect_gpio_out(dev, 4, qdev_get_gpio_in(i2c_dev, 1));



    for (i = 0; i < 3; i++) {

        qdev_connect_gpio_out(dev, i, qdev_get_gpio_in(lcd_dev, i));

    }

    for (i = 0; i < 4; i++) {

        qdev_connect_gpio_out(key_dev, i, qdev_get_gpio_in(dev, i + 8));

    }

    for (i = 4; i < 8; i++) {

        qdev_connect_gpio_out(key_dev, i, qdev_get_gpio_in(dev, i + 15));

    }



    wm8750_dev = i2c_create_slave(i2c, "wm8750", MP_WM_ADDR);

    dev = qdev_create(NULL, "mv88w8618_audio");

    s = SYS_BUS_DEVICE(dev);

    qdev_prop_set_ptr(dev, "wm8750", wm8750_dev);

    qdev_init_nofail(dev);

    sysbus_mmio_map(s, 0, MP_AUDIO_BASE);

    sysbus_connect_irq(s, 0, pic[MP_AUDIO_IRQ]);



    musicpal_binfo.ram_size = MP_RAM_DEFAULT_SIZE;

    musicpal_binfo.kernel_filename = kernel_filename;

    musicpal_binfo.kernel_cmdline = kernel_cmdline;

    musicpal_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(cpu, &musicpal_binfo);

}
