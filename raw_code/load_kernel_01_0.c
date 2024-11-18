void load_kernel (CPUState *env, int ram_size, const char *kernel_filename,

		  const char *kernel_cmdline,

		  const char *initrd_filename)

{

    int64_t entry = 0;

    long kernel_size, initrd_size;



    kernel_size = load_elf(kernel_filename, VIRT_TO_PHYS_ADDEND, &entry);

    if (kernel_size >= 0) {

        if ((entry & ~0x7fffffffULL) == 0x80000000)

            entry = (int32_t)entry;

        env->PC = entry;

    } else {

        kernel_size = load_image(kernel_filename,

                                 phys_ram_base + KERNEL_LOAD_ADDR + VIRT_TO_PHYS_ADDEND);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

        env->PC = KERNEL_LOAD_ADDR;

    }



    /* load initrd */

    initrd_size = 0;

    if (initrd_filename) {

        initrd_size = load_image(initrd_filename,

                                 phys_ram_base + INITRD_LOAD_ADDR + VIRT_TO_PHYS_ADDEND);

        if (initrd_size == (target_ulong) -1) {

            fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                    initrd_filename);

            exit(1);

        }

    }



    /* Store command line.  */

    if (initrd_size > 0) {

        int ret;

        ret = sprintf(phys_ram_base + (16 << 20) - 256,

                      "rd_start=0x" TLSZ " rd_size=%li ",

                      INITRD_LOAD_ADDR,

                      initrd_size);

        strcpy (phys_ram_base + (16 << 20) - 256 + ret, kernel_cmdline);

    }

    else {

        strcpy (phys_ram_base + (16 << 20) - 256, kernel_cmdline);

    }



    *(int *)(phys_ram_base + (16 << 20) - 260) = tswap32 (0x12345678);

    *(int *)(phys_ram_base + (16 << 20) - 264) = tswap32 (ram_size);

}
