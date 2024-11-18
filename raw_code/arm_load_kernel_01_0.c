void arm_load_kernel(ARMCPU *cpu, struct arm_boot_info *info)

{

    CPUARMState *env = &cpu->env;

    int kernel_size;

    int initrd_size;

    int n;

    int is_linux = 0;

    uint64_t elf_entry;

    target_phys_addr_t entry;

    int big_endian;

    QemuOpts *machine_opts;



    /* Load the kernel.  */

    if (!info->kernel_filename) {

        fprintf(stderr, "Kernel image must be specified\n");

        exit(1);

    }



    machine_opts = qemu_opts_find(qemu_find_opts("machine"), 0);

    if (machine_opts) {

        info->dtb_filename = qemu_opt_get(machine_opts, "dtb");

    } else {

        info->dtb_filename = NULL;

    }



    if (!info->secondary_cpu_reset_hook) {

        info->secondary_cpu_reset_hook = default_reset_secondary;

    }

    if (!info->write_secondary_boot) {

        info->write_secondary_boot = default_write_secondary;

    }



    if (info->nb_cpus == 0)

        info->nb_cpus = 1;



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif



    /* Assume that raw images are linux kernels, and ELF images are not.  */

    kernel_size = load_elf(info->kernel_filename, NULL, NULL, &elf_entry,

                           NULL, NULL, big_endian, ELF_MACHINE, 1);

    entry = elf_entry;

    if (kernel_size < 0) {

        kernel_size = load_uimage(info->kernel_filename, &entry, NULL,

                                  &is_linux);

    }

    if (kernel_size < 0) {

        entry = info->loader_start + KERNEL_LOAD_ADDR;

        kernel_size = load_image_targphys(info->kernel_filename, entry,

                                          info->ram_size - KERNEL_LOAD_ADDR);

        is_linux = 1;

    }

    if (kernel_size < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n",

                info->kernel_filename);

        exit(1);

    }

    info->entry = entry;

    if (is_linux) {

        if (info->initrd_filename) {

            initrd_size = load_image_targphys(info->initrd_filename,

                                              info->loader_start

                                              + INITRD_LOAD_ADDR,

                                              info->ram_size

                                              - INITRD_LOAD_ADDR);

            if (initrd_size < 0) {

                fprintf(stderr, "qemu: could not load initrd '%s'\n",

                        info->initrd_filename);

                exit(1);

            }

        } else {

            initrd_size = 0;

        }

        info->initrd_size = initrd_size;



        bootloader[4] = info->board_id;



        /* for device tree boot, we pass the DTB directly in r2. Otherwise

         * we point to the kernel args.

         */

        if (info->dtb_filename) {

            /* Place the DTB after the initrd in memory */

            target_phys_addr_t dtb_start = TARGET_PAGE_ALIGN(info->loader_start

                                                             + INITRD_LOAD_ADDR

                                                             + initrd_size);

            if (load_dtb(dtb_start, info)) {

                exit(1);

            }

            bootloader[5] = dtb_start;

        } else {

            bootloader[5] = info->loader_start + KERNEL_ARGS_ADDR;

            if (info->ram_size >= (1ULL << 32)) {

                fprintf(stderr, "qemu: RAM size must be less than 4GB to boot"

                        " Linux kernel using ATAGS (try passing a device tree"

                        " using -dtb)\n");

                exit(1);

            }

        }

        bootloader[6] = entry;

        for (n = 0; n < sizeof(bootloader) / 4; n++) {

            bootloader[n] = tswap32(bootloader[n]);

        }

        rom_add_blob_fixed("bootloader", bootloader, sizeof(bootloader),

                           info->loader_start);

        if (info->nb_cpus > 1) {

            info->write_secondary_boot(cpu, info);

        }

    }

    info->is_linux = is_linux;



    for (; env; env = env->next_cpu) {

        cpu = arm_env_get_cpu(env);

        env->boot_info = info;

        qemu_register_reset(do_cpu_reset, cpu);

    }

}
