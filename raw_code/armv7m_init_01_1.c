qemu_irq *armv7m_init(MemoryRegion *system_memory, int mem_size, int num_irq,

                      const char *kernel_filename, const char *cpu_model)

{

    ARMCPU *cpu;

    CPUARMState *env;

    DeviceState *nvic;

    qemu_irq *pic = g_new(qemu_irq, num_irq);

    int image_size;

    uint64_t entry;

    uint64_t lowaddr;

    int i;

    int big_endian;

    MemoryRegion *hack = g_new(MemoryRegion, 1);



    if (cpu_model == NULL) {

	cpu_model = "cortex-m3";

    }

    cpu = cpu_arm_init(cpu_model);

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    armv7m_bitband_init();



    nvic = qdev_create(NULL, "armv7m_nvic");

    qdev_prop_set_uint32(nvic, "num-irq", num_irq);

    env->nvic = nvic;

    qdev_init_nofail(nvic);

    sysbus_connect_irq(SYS_BUS_DEVICE(nvic), 0,

                       qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_IRQ));

    for (i = 0; i < num_irq; i++) {

        pic[i] = qdev_get_gpio_in(nvic, i);

    }



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif



    if (!kernel_filename && !qtest_enabled()) {

        fprintf(stderr, "Guest image must be specified (using -kernel)\n");

        exit(1);

    }



    if (kernel_filename) {

        image_size = load_elf(kernel_filename, NULL, NULL, &entry, &lowaddr,

                              NULL, big_endian, ELF_MACHINE, 1);

        if (image_size < 0) {

            image_size = load_image_targphys(kernel_filename, 0, mem_size);

            lowaddr = 0;

        }

        if (image_size < 0) {

            error_report("Could not load kernel '%s'", kernel_filename);

            exit(1);

        }

    }



    /* Hack to map an additional page of ram at the top of the address

       space.  This stops qemu complaining about executing code outside RAM

       when returning from an exception.  */

    memory_region_init_ram(hack, NULL, "armv7m.hack", 0x1000, &error_abort);

    vmstate_register_ram_global(hack);

    memory_region_add_subregion(system_memory, 0xfffff000, hack);



    qemu_register_reset(armv7m_reset, cpu);

    return pic;

}
