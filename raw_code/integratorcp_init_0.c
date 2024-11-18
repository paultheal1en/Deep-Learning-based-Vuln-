static void integratorcp_init(ram_addr_t ram_size, int vga_ram_size,

                     const char *boot_device, DisplayState *ds,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    uint32_t ram_offset;

    qemu_irq *pic;

    qemu_irq *cpu_pic;

    int sd;



    if (!cpu_model)

        cpu_model = "arm926";

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    ram_offset = qemu_ram_alloc(ram_size);

    /* ??? On a real system the first 1Mb is mapped as SSRAM or boot flash.  */

    /* ??? RAM should repeat to fill physical memory space.  */

    /* SDRAM at address zero*/

    cpu_register_physical_memory(0, ram_size, ram_offset | IO_MEM_RAM);

    /* And again at address 0x80000000 */

    cpu_register_physical_memory(0x80000000, ram_size, ram_offset | IO_MEM_RAM);



    integratorcm_init(ram_size >> 20);

    cpu_pic = arm_pic_init_cpu(env);

    pic = icp_pic_init(0x14000000, cpu_pic[ARM_PIC_CPU_IRQ],

                       cpu_pic[ARM_PIC_CPU_FIQ]);

    icp_pic_init(0xca000000, pic[26], NULL);

    icp_pit_init(0x13000000, pic, 5);

    pl031_init(0x15000000, pic[8]);

    pl011_init(0x16000000, pic[1], serial_hds[0], PL011_ARM);

    pl011_init(0x17000000, pic[2], serial_hds[1], PL011_ARM);

    icp_control_init(0xcb000000);

    pl050_init(0x18000000, pic[3], 0);

    pl050_init(0x19000000, pic[4], 1);

    sd = drive_get_index(IF_SD, 0, 0);

    if (sd == -1) {

        fprintf(stderr, "qemu: missing SecureDigital card\n");

        exit(1);

    }

    pl181_init(0x1c000000, drives_table[sd].bdrv, pic[23], pic[24]);

    if (nd_table[0].vlan) {

        if (nd_table[0].model == NULL

            || strcmp(nd_table[0].model, "smc91c111") == 0) {

            smc91c111_init(&nd_table[0], 0xc8000000, pic[27]);

        } else if (strcmp(nd_table[0].model, "?") == 0) {

            fprintf(stderr, "qemu: Supported NICs: smc91c111\n");

            exit (1);

        } else {

            fprintf(stderr, "qemu: Unsupported NIC: %s\n", nd_table[0].model);

            exit (1);

        }

    }

    pl110_init(ds, 0xc0000000, pic[22], 0);



    integrator_binfo.ram_size = ram_size;

    integrator_binfo.kernel_filename = kernel_filename;

    integrator_binfo.kernel_cmdline = kernel_cmdline;

    integrator_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(env, &integrator_binfo);

}
