static void s390_init(ram_addr_t ram_size,

                      const char *boot_device,

                      const char *kernel_filename,

                      const char *kernel_cmdline,

                      const char *initrd_filename,

                      const char *cpu_model)

{

    CPUState *env = NULL;

    ram_addr_t ram_addr;

    ram_addr_t kernel_size = 0;

    ram_addr_t initrd_offset;

    ram_addr_t initrd_size = 0;

    int i;



    /* XXX we only work on KVM for now */



    if (!kvm_enabled()) {

        fprintf(stderr, "The S390 target only works with KVM enabled\n");

        exit(1);

    }



    /* get a BUS */

    s390_bus = s390_virtio_bus_init(&ram_size);



    /* allocate RAM */

    ram_addr = qemu_ram_alloc(NULL, "s390.ram", ram_size);

    cpu_register_physical_memory(0, ram_size, ram_addr);



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "host";

    }



    ipi_states = qemu_malloc(sizeof(CPUState *) * smp_cpus);



    for (i = 0; i < smp_cpus; i++) {

        CPUState *tmp_env;



        tmp_env = cpu_init(cpu_model);

        if (!env) {

            env = tmp_env;

        }

        ipi_states[i] = tmp_env;

        tmp_env->halted = 1;

        tmp_env->exception_index = EXCP_HLT;

    }



    env->halted = 0;

    env->exception_index = 0;



    if (kernel_filename) {

        kernel_size = load_image(kernel_filename, qemu_get_ram_ptr(0));



        if (lduw_phys(KERN_IMAGE_START) != 0x0dd0) {

            fprintf(stderr, "Specified image is not an s390 boot image\n");

            exit(1);

        }



        env->psw.addr = KERN_IMAGE_START;

        env->psw.mask = 0x0000000180000000ULL;

    } else {

        ram_addr_t bios_size = 0;

        char *bios_filename;



        /* Load zipl bootloader */

        if (bios_name == NULL) {

            bios_name = ZIPL_FILENAME;

        }



        bios_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        bios_size = load_image(bios_filename, qemu_get_ram_ptr(ZIPL_LOAD_ADDR));




        if ((long)bios_size < 0) {

            hw_error("could not load bootloader '%s'\n", bios_name);

        }



        if (bios_size > 4096) {

            hw_error("stage1 bootloader is > 4k\n");

        }



        env->psw.addr = ZIPL_START;

        env->psw.mask = 0x0000000180000000ULL;

    }



    if (initrd_filename) {

        initrd_offset = INITRD_START;

        while (kernel_size + 0x100000 > initrd_offset) {

            initrd_offset += 0x100000;

        }

        initrd_size = load_image(initrd_filename, qemu_get_ram_ptr(initrd_offset));



        stq_phys(INITRD_PARM_START, initrd_offset);

        stq_phys(INITRD_PARM_SIZE, initrd_size);

    }



    if (kernel_cmdline) {

        cpu_physical_memory_rw(KERN_PARM_AREA, (uint8_t *)kernel_cmdline,

                               strlen(kernel_cmdline), 1);

    }



    /* Create VirtIO network adapters */

    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];

        DeviceState *dev;



        if (!nd->model) {

            nd->model = qemu_strdup("virtio");

        }



        if (strcmp(nd->model, "virtio")) {

            fprintf(stderr, "S390 only supports VirtIO nics\n");

            exit(1);

        }



        dev = qdev_create((BusState *)s390_bus, "virtio-net-s390");

        qdev_set_nic_properties(dev, nd);

        qdev_init_nofail(dev);

    }



    /* Create VirtIO disk drives */

    for(i = 0; i < MAX_BLK_DEVS; i++) {

        DriveInfo *dinfo;

        DeviceState *dev;



        dinfo = drive_get(IF_IDE, 0, i);

        if (!dinfo) {

            continue;

        }



        dev = qdev_create((BusState *)s390_bus, "virtio-blk-s390");

        qdev_prop_set_drive_nofail(dev, "drive", dinfo->bdrv);

        qdev_init_nofail(dev);

    }

}