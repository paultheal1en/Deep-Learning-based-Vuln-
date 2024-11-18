static void sun4m_common_init(int ram_size, int boot_device, DisplayState *ds,

                              const char *kernel_filename, const char *kernel_cmdline,

                              const char *initrd_filename, const char *cpu_model,

                              unsigned int machine, int max_ram)

{

    if ((unsigned int)ram_size > (unsigned int)max_ram) {

        fprintf(stderr, "qemu: Too much memory for this machine: %d, maximum %d\n",

                (unsigned int)ram_size / (1024 * 1024),

                (unsigned int)max_ram / (1024 * 1024));

        exit(1);

    }

    sun4m_hw_init(&hwdefs[machine], ram_size, ds, cpu_model);



    sun4m_load_kernel(hwdefs[machine].vram_size, ram_size, boot_device,

                      kernel_filename, kernel_cmdline, initrd_filename,

                      hwdefs[machine].machine_id);

}
