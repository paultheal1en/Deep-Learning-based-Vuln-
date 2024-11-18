static int s390_ipl_init(SysBusDevice *dev)

{

    S390IPLState *ipl = S390_IPL(dev);

    int kernel_size;



    if (!ipl->kernel) {

        int bios_size;

        char *bios_filename;



        /* Load zipl bootloader */

        if (bios_name == NULL) {

            bios_name = ipl->firmware;

        }



        bios_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (bios_filename == NULL) {

            hw_error("could not find stage1 bootloader\n");

        }



        bios_size = load_elf(bios_filename, NULL, NULL, &ipl->start_addr, NULL,

                             NULL, 1, ELF_MACHINE, 0);

        if (bios_size == -1) {

            bios_size = load_image_targphys(bios_filename, ZIPL_IMAGE_START,

                                            4096);

            ipl->start_addr = ZIPL_IMAGE_START;

            if (bios_size > 4096) {

                hw_error("stage1 bootloader is > 4k\n");

            }

        }

        g_free(bios_filename);



        if (bios_size == -1) {

            hw_error("could not load bootloader '%s'\n", bios_name);

        }

        return 0;

    } else {

        uint64_t pentry = KERN_IMAGE_START;

        kernel_size = load_elf(ipl->kernel, NULL, NULL, &pentry, NULL,

                               NULL, 1, ELF_MACHINE, 0);

        if (kernel_size == -1) {

            kernel_size = load_image_targphys(ipl->kernel, 0, ram_size);

        }

        if (kernel_size == -1) {

            fprintf(stderr, "could not load kernel '%s'\n", ipl->kernel);

            return -1;

        }

        /*

         * Is it a Linux kernel (starting at 0x10000)? If yes, we fill in the

         * kernel parameters here as well. Note: For old kernels (up to 3.2)

         * we can not rely on the ELF entry point - it was 0x800 (the SALIPL

         * loader) and it won't work. For this case we force it to 0x10000, too.

         */

        if (pentry == KERN_IMAGE_START || pentry == 0x800) {

            ipl->start_addr = KERN_IMAGE_START;

            /* Overwrite parameters in the kernel image, which are "rom" */

            strcpy(rom_ptr(KERN_PARM_AREA), ipl->cmdline);

        } else {

            ipl->start_addr = pentry;

        }

    }

    if (ipl->initrd) {

        ram_addr_t initrd_offset;

        int initrd_size;



        initrd_offset = INITRD_START;

        while (kernel_size + 0x100000 > initrd_offset) {

            initrd_offset += 0x100000;

        }

        initrd_size = load_image_targphys(ipl->initrd, initrd_offset,

                                          ram_size - initrd_offset);

        if (initrd_size == -1) {

            fprintf(stderr, "qemu: could not load initrd '%s'\n", ipl->initrd);

            exit(1);

        }



        /* we have to overwrite values in the kernel image, which are "rom" */

        stq_p(rom_ptr(INITRD_PARM_START), initrd_offset);

        stq_p(rom_ptr(INITRD_PARM_SIZE), initrd_size);

    }



    return 0;

}
