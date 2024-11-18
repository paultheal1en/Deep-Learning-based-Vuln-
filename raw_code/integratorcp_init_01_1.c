static void integratorcp_init(ram_addr_t ram_size,

                     const char *boot_device,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename, const char *cpu_model)

{

    CPUState *env;

    ram_addr_t ram_offset;

    qemu_irq pic[32];

    qemu_irq *cpu_pic;

    DeviceState *dev;

    int i;



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



    dev = qdev_create(NULL, "integrator_core");

    qdev_prop_set_uint32(dev, "memsz", ram_size >> 20);

    qdev_init(dev);

    sysbus_mmio_map((SysBusDevice *)dev, 0, 0x10000000);



    cpu_pic = arm_pic_init_cpu(env);

    dev = sysbus_create_varargs("integrator_pic", 0x14000000,

                                cpu_pic[ARM_PIC_CPU_IRQ],

                                cpu_pic[ARM_PIC_CPU_FIQ], NULL);

    for (i = 0; i < 32; i++) {

        pic[i] = qdev_get_gpio_in(dev, i);

    }

    sysbus_create_simple("integrator_pic", 0xca000000, pic[26]);

    sysbus_create_varargs("integrator_pit", 0x13000000,

                          pic[5], pic[6], pic[7], NULL);

    sysbus_create_simple("pl031", 0x15000000, pic[8]);

    sysbus_create_simple("pl011", 0x16000000, pic[1]);

    sysbus_create_simple("pl011", 0x17000000, pic[2]);

    icp_control_init(0xcb000000);

    sysbus_create_simple("pl050_keyboard", 0x18000000, pic[3]);

    sysbus_create_simple("pl050_mouse", 0x19000000, pic[4]);

    sysbus_create_varargs("pl181", 0x1c000000, pic[23], pic[24], NULL);

    if (nd_table[0].vlan)

        smc91c111_init(&nd_table[0], 0xc8000000, pic[27]);



    sysbus_create_simple("pl110", 0xc0000000, pic[22]);



    integrator_binfo.ram_size = ram_size;

    integrator_binfo.kernel_filename = kernel_filename;

    integrator_binfo.kernel_cmdline = kernel_cmdline;

    integrator_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(env, &integrator_binfo);

}
