static void musicpal_init(ram_addr_t ram_size,

               const char *boot_device,

               const char *kernel_filename, const char *kernel_cmdline,

               const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    qemu_irq *cpu_pic;

    qemu_irq pic[32];

    DeviceState *dev;

    DeviceState *i2c_dev;

    DeviceState *lcd_dev;

    DeviceState *key_dev;

#ifdef HAS_AUDIO

    DeviceState *wm8750_dev;

    SysBusDevice *s;

#endif

    i2c_bus *i2c;

    int i;

    unsigned long flash_size;

    DriveInfo *dinfo;

    ram_addr_t sram_off;



    if (!cpu_model) {

        cpu_model = "arm926";

    }

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    cpu_pic = arm_pic_init_cpu(env);



    /* For now we use a fixed - the original - RAM size */

    cpu_register_physical_memory(0, MP_RAM_DEFAULT_SIZE,

                                 qemu_ram_alloc(MP_RAM_DEFAULT_SIZE));



    sram_off = qemu_ram_alloc(MP_SRAM_SIZE);

    cpu_register_physical_memory(MP_SRAM_BASE, MP_SRAM_SIZE, sram_off);



    dev = sysbus_create_simple("mv88w8618_pic", MP_PIC_BASE,

                               cpu_pic[ARM_PIC_CPU_IRQ]);

    for (i = 0; i < 32; i++) {

        pic[i] = qdev_get_gpio_in(dev, i);

    }

    sysbus_create_varargs("mv88w8618_pit", MP_PIT_BASE, pic[MP_TIMER1_IRQ],

                          pic[MP_TIMER2_IRQ], pic[MP_TIMER3_IRQ],

                          pic[MP_TIMER4_IRQ], NULL);



    if (serial_hds[0]) {

        serial_mm_init(MP_UART1_BASE, 2, pic[MP_UART1_IRQ], 1825000,

                   serial_hds[0], 1);

    }

    if (serial_hds[1]) {

        serial_mm_init(MP_UART2_BASE, 2, pic[MP_UART2_IRQ], 1825000,

                   serial_hds[1], 1);

    }



    /* Register flash */

    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (dinfo) {

        flash_size = bdrv_getlength(dinfo->bdrv);

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

        pflash_cfi02_register(0-MP_FLASH_SIZE_MAX, qemu_ram_alloc(flash_size),

                              dinfo->bdrv, 0x10000,

                              (flash_size + 0xffff) >> 16,

                              MP_FLASH_SIZE_MAX / flash_size,

                              2, 0x00BF, 0x236D, 0x0000, 0x0000,

                              0x5555, 0x2AAA);

    }

    sysbus_create_simple("mv88w8618_flashcfg", MP_FLASHCFG_BASE, NULL);



    qemu_check_nic_model(&nd_table[0], "mv88w8618");

    dev = qdev_create(NULL, "mv88w8618_eth");

    dev->nd = &nd_table[0];

    qdev_init(dev);

    sysbus_mmio_map(sysbus_from_qdev(dev), 0, MP_ETH_BASE);

    sysbus_connect_irq(sysbus_from_qdev(dev), 0, pic[MP_ETH_IRQ]);



    sysbus_create_simple("mv88w8618_wlan", MP_WLAN_BASE, NULL);



    musicpal_misc_init();



    dev = sysbus_create_simple("musicpal_gpio", MP_GPIO_BASE, pic[MP_GPIO_IRQ]);

    i2c_dev = sysbus_create_simple("bitbang_i2c", 0, NULL);

    i2c = (i2c_bus *)qdev_get_child_bus(i2c_dev, "i2c");



    lcd_dev = sysbus_create_simple("musicpal_lcd", MP_LCD_BASE, NULL);

    key_dev = sysbus_create_simple("musicpal_key", 0, NULL);



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



#ifdef HAS_AUDIO

    wm8750_dev = i2c_create_slave(i2c, "wm8750", MP_WM_ADDR);

    dev = qdev_create(NULL, "mv88w8618_audio");

    s = sysbus_from_qdev(dev);

    qdev_prop_set_ptr(dev, "wm8750", wm8750_dev);

    qdev_init(dev);

    sysbus_mmio_map(s, 0, MP_AUDIO_BASE);

    sysbus_connect_irq(s, 0, pic[MP_AUDIO_IRQ]);

#endif



    musicpal_binfo.ram_size = MP_RAM_DEFAULT_SIZE;

    musicpal_binfo.kernel_filename = kernel_filename;

    musicpal_binfo.kernel_cmdline = kernel_cmdline;

    musicpal_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(env, &musicpal_binfo);

}
