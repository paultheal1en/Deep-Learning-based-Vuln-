static void n8x0_init(QEMUMachineInitArgs *args,

                      struct arm_boot_info *binfo, int model)

{

    MemoryRegion *sysmem = get_system_memory();

    struct n800_s *s = (struct n800_s *) g_malloc0(sizeof(*s));

    int sdram_size = binfo->ram_size;



    s->mpu = omap2420_mpu_init(sysmem, sdram_size, args->cpu_model);



    /* Setup peripherals

     *

     * Believed external peripherals layout in the N810:

     * (spi bus 1)

     *   tsc2005

     *   lcd_mipid

     * (spi bus 2)

     *   Conexant cx3110x (WLAN)

     *   optional: pc2400m (WiMAX)

     * (i2c bus 0)

     *   TLV320AIC33 (audio codec)

     *   TCM825x (camera by Toshiba)

     *   lp5521 (clever LEDs)

     *   tsl2563 (light sensor, hwmon, model 7, rev. 0)

     *   lm8323 (keypad, manf 00, rev 04)

     * (i2c bus 1)

     *   tmp105 (temperature sensor, hwmon)

     *   menelaus (pm)

     * (somewhere on i2c - maybe N800-only)

     *   tea5761 (FM tuner)

     * (serial 0)

     *   GPS

     * (some serial port)

     *   csr41814 (Bluetooth)

     */

    n8x0_gpio_setup(s);

    n8x0_nand_setup(s);

    n8x0_i2c_setup(s);

    if (model == 800)

        n800_tsc_kbd_setup(s);

    else if (model == 810) {

        n810_tsc_setup(s);

        n810_kbd_setup(s);

    }

    n8x0_spi_setup(s);

    n8x0_dss_setup(s);

    n8x0_cbus_setup(s);

    n8x0_uart_setup(s);

    if (usb_enabled(false)) {

        n8x0_usb_setup(s);

    }



    if (args->kernel_filename) {

        /* Or at the linux loader.  */

        binfo->kernel_filename = args->kernel_filename;

        binfo->kernel_cmdline = args->kernel_cmdline;

        binfo->initrd_filename = args->initrd_filename;

        arm_load_kernel(s->mpu->cpu, binfo);



        qemu_register_reset(n8x0_boot_init, s);

    }



    if (option_rom[0].name &&

        (args->boot_device[0] == 'n' || !args->kernel_filename)) {

        uint8_t nolo_tags[0x10000];

        /* No, wait, better start at the ROM.  */

        s->mpu->cpu->env.regs[15] = OMAP2_Q2_BASE + 0x400000;



        /* This is intended for loading the `secondary.bin' program from

         * Nokia images (the NOLO bootloader).  The entry point seems

         * to be at OMAP2_Q2_BASE + 0x400000.

         *

         * The `2nd.bin' files contain some kind of earlier boot code and

         * for them the entry point needs to be set to OMAP2_SRAM_BASE.

         *

         * The code above is for loading the `zImage' file from Nokia

         * images.  */

        load_image_targphys(option_rom[0].name,

                            OMAP2_Q2_BASE + 0x400000,

                            sdram_size - 0x400000);



        n800_setup_nolo_tags(nolo_tags);

        cpu_physical_memory_write(OMAP2_SRAM_BASE, nolo_tags, 0x10000);

    }

}
