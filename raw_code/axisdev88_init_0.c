void axisdev88_init (ram_addr_t ram_size, int vga_ram_size,

                     const char *boot_device, DisplayState *ds,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    struct etraxfs_pic *pic;

    void *etraxfs_dmac;

    struct etraxfs_dma_client *eth[2] = {NULL, NULL};

    int kernel_size;

    int i;

    int nand_regs;

    int gpio_regs;

    ram_addr_t phys_ram;

    ram_addr_t phys_intmem;



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "crisv32";

    }

    env = cpu_init(cpu_model);

    qemu_register_reset(main_cpu_reset, env);



    /* allocate RAM */

    phys_ram = qemu_ram_alloc(ram_size);

    cpu_register_physical_memory(0x40000000, ram_size, phys_ram | IO_MEM_RAM);



    /* The ETRAX-FS has 128Kb on chip ram, the docs refer to it as the 

       internal memory.  */

    phys_intmem = qemu_ram_alloc(INTMEM_SIZE);

    cpu_register_physical_memory(0x38000000, INTMEM_SIZE,

                                 phys_intmem | IO_MEM_RAM);





      /* Attach a NAND flash to CS1.  */

    nand_state.nand = nand_init(NAND_MFR_STMICRO, 0x39);

    nand_regs = cpu_register_io_memory(0, nand_read, nand_write, &nand_state);

    cpu_register_physical_memory(0x10000000, 0x05000000, nand_regs);



    gpio_state.nand = &nand_state;

    gpio_regs = cpu_register_io_memory(0, gpio_read, gpio_write, &gpio_state);

    cpu_register_physical_memory(0x3001a000, 0x5c, gpio_regs);





    pic = etraxfs_pic_init(env, 0x3001c000);

    etraxfs_dmac = etraxfs_dmac_init(env, 0x30000000, 10);

    for (i = 0; i < 10; i++) {

        /* On ETRAX, odd numbered channels are inputs.  */

        etraxfs_dmac_connect(etraxfs_dmac, i, pic->irq + 7 + i, i & 1);

    }



    /* Add the two ethernet blocks.  */

    eth[0] = etraxfs_eth_init(&nd_table[0], env, pic->irq + 25, 0x30034000);

    if (nb_nics > 1)

        eth[1] = etraxfs_eth_init(&nd_table[1], env, pic->irq + 26, 0x30036000);



    /* The DMA Connector block is missing, hardwire things for now.  */

    etraxfs_dmac_connect_client(etraxfs_dmac, 0, eth[0]);

    etraxfs_dmac_connect_client(etraxfs_dmac, 1, eth[0] + 1);

    if (eth[1]) {

        etraxfs_dmac_connect_client(etraxfs_dmac, 6, eth[1]);

        etraxfs_dmac_connect_client(etraxfs_dmac, 7, eth[1] + 1);

    }



    /* 2 timers.  */

    etraxfs_timer_init(env, pic->irq + 0x1b, pic->nmi + 1, 0x3001e000);

    etraxfs_timer_init(env, pic->irq + 0x1b, pic->nmi + 1, 0x3005e000);



    for (i = 0; i < 4; i++) {

        if (serial_hds[i]) {

            etraxfs_ser_init(env, pic->irq + 0x14 + i,

                             serial_hds[i], 0x30026000 + i * 0x2000);

        }

    }



    if (kernel_filename) {

        uint64_t entry, high;

        int kcmdline_len;



        /* Boots a kernel elf binary, os/linux-2.6/vmlinux from the axis 

           devboard SDK.  */

        kernel_size = load_elf(kernel_filename, -0x80000000LL,

                               &entry, NULL, &high);

        bootstrap_pc = entry;

        if (kernel_size < 0) {

            /* Takes a kimage from the axis devboard SDK.  */

            kernel_size = load_image(kernel_filename, phys_ram_base + 0x4000);

            bootstrap_pc = 0x40004000;

            env->regs[9] = 0x40004000 + kernel_size;

        }

        env->regs[8] = 0x56902387; /* RAM init magic.  */



        if (kernel_cmdline && (kcmdline_len = strlen(kernel_cmdline))) {

            if (kcmdline_len > 256) {

                fprintf(stderr, "Too long CRIS kernel cmdline (max 256)\n");

                exit(1);

            }

            pstrcpy_targphys(high, 256, kernel_cmdline);

            /* Let the kernel know we are modifying the cmdline.  */

            env->regs[10] = 0x87109563;

            env->regs[11] = high;

        }

    }

    env->pc = bootstrap_pc;



    printf ("pc =%x\n", env->pc);

    printf ("ram size =%ld\n", ram_size);

}
