static void cpu_openrisc_load_kernel(ram_addr_t ram_size,

                                     const char *kernel_filename,

                                     OpenRISCCPU *cpu)

{

    long kernel_size;

    uint64_t elf_entry;

    hwaddr entry;



    if (kernel_filename && !qtest_enabled()) {

        kernel_size = load_elf(kernel_filename, NULL, NULL,

                               &elf_entry, NULL, NULL, 1, ELF_MACHINE, 1);

        entry = elf_entry;

        if (kernel_size < 0) {

            kernel_size = load_uimage(kernel_filename,

                                      &entry, NULL, NULL);

        }

        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename,

                                              KERNEL_LOAD_ADDR,

                                              ram_size - KERNEL_LOAD_ADDR);

            entry = KERNEL_LOAD_ADDR;

        }



        if (kernel_size < 0) {

            fprintf(stderr, "QEMU: couldn't load the kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

    }



    cpu->env.pc = entry;

}
