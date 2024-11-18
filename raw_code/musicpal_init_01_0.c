static void musicpal_init(ram_addr_t ram_size, int vga_ram_size,

               const char *boot_device, DisplayState *ds,

               const char *kernel_filename, const char *kernel_cmdline,

               const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    qemu_irq *pic;

    int index;

    int iomemtype;

    unsigned long flash_size;



    if (!cpu_model)

        cpu_model = "arm926";



    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    pic = arm_pic_init_cpu(env);



    /* For now we use a fixed - the original - RAM size */

    cpu_register_physical_memory(0, MP_RAM_DEFAULT_SIZE,

                                 qemu_ram_alloc(MP_RAM_DEFAULT_SIZE));



    sram_off = qemu_ram_alloc(MP_SRAM_SIZE);

    cpu_register_physical_memory(MP_SRAM_BASE, MP_SRAM_SIZE, sram_off);



    /* Catch various stuff not handled by separate subsystems */

    iomemtype = cpu_register_io_memory(0, musicpal_readfn,

                                       musicpal_writefn, env);

    cpu_register_physical_memory(0x80000000, 0x10000, iomemtype);



    pic = mv88w8618_pic_init(MP_PIC_BASE, pic[ARM_PIC_CPU_IRQ]);

    mv88w8618_pit_init(MP_PIT_BASE, pic, MP_TIMER1_IRQ);



    if (serial_hds[0])

        serial_mm_init(MP_UART1_BASE, 2, pic[MP_UART1_IRQ], 1825000,

                   serial_hds[0], 1);

    if (serial_hds[1])

        serial_mm_init(MP_UART2_BASE, 2, pic[MP_UART2_IRQ], 1825000,

                   serial_hds[1], 1);



    /* Register flash */

    index = drive_get_index(IF_PFLASH, 0, 0);

    if (index != -1) {

        flash_size = bdrv_getlength(drives_table[index].bdrv);

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

                              drives_table[index].bdrv, 0x10000,

                              (flash_size + 0xffff) >> 16,

                              MP_FLASH_SIZE_MAX / flash_size,

                              2, 0x00BF, 0x236D, 0x0000, 0x0000,

                              0x5555, 0x2AAA);

    }

    mv88w8618_flashcfg_init(MP_FLASHCFG_BASE);



    musicpal_lcd_init(ds, MP_LCD_BASE);



    qemu_add_kbd_event_handler(musicpal_key_event, pic[MP_GPIO_IRQ]);



    /*

     * Wait a bit to catch menu button during U-Boot start-up

     * (to trigger emergency update).

     */

    sleep(1);



    mv88w8618_eth_init(&nd_table[0], MP_ETH_BASE, pic[MP_ETH_IRQ]);



    mixer_i2c = musicpal_audio_init(MP_AUDIO_BASE, pic[MP_AUDIO_IRQ]);



    musicpal_binfo.ram_size = MP_RAM_DEFAULT_SIZE;

    musicpal_binfo.kernel_filename = kernel_filename;

    musicpal_binfo.kernel_cmdline = kernel_cmdline;

    musicpal_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(env, &musicpal_binfo);

}
