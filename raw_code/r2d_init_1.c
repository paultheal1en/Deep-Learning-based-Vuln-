static void r2d_init(ram_addr_t ram_size, int vga_ram_size,

              const char *boot_device,

	      const char *kernel_filename, const char *kernel_cmdline,

	      const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    struct SH7750State *s;

    ram_addr_t sdram_addr, sm501_vga_ram_addr;

    qemu_irq *irq;

    PCIBus *pci;

    int i;



    if (!cpu_model)

        cpu_model = "SH7751R";



    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



    /* Allocate memory space */

    sdram_addr = qemu_ram_alloc(SDRAM_SIZE);

    cpu_register_physical_memory(SDRAM_BASE, SDRAM_SIZE, sdram_addr);

    /* Register peripherals */

    s = sh7750_init(env);

    irq = r2d_fpga_init(0x04000000, sh7750_irl(s));

    pci = sh_pci_register_bus(r2d_pci_set_irq, r2d_pci_map_irq, irq, 0, 4);



    sm501_vga_ram_addr = qemu_ram_alloc(SM501_VRAM_SIZE);

    sm501_init(0x10000000, sm501_vga_ram_addr, SM501_VRAM_SIZE,

	       serial_hds[2]);



    /* onboard CF (True IDE mode, Master only). */

    mmio_ide_init(0x14001000, 0x1400080c, irq[CF_IDE], 1,

        drives_table[drive_get_index(IF_IDE, 0, 0)].bdrv, NULL);



    /* NIC: rtl8139 on-board, and 2 slots. */

    pci_nic_init(pci, &nd_table[0], 2 << 3, "rtl8139");

    for (i = 1; i < nb_nics; i++)

        pci_nic_init(pci, &nd_table[i], -1, "ne2k_pci");



    /* Todo: register on board registers */

    {

      int kernel_size;

      /* initialization which should be done by firmware */

      stl_phys(SH7750_BCR1, 1<<3); /* cs3 SDRAM */

      stw_phys(SH7750_BCR2, 3<<(3*2)); /* cs3 32bit */



      kernel_size = load_image(kernel_filename, phys_ram_base);



      if (kernel_size < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n", kernel_filename);

        exit(1);

      }



      env->pc = SDRAM_BASE | 0xa0000000; /* Start from P2 area */

    }

}
