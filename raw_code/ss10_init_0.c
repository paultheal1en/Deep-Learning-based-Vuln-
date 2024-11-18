static void ss10_init(int ram_size, int vga_ram_size, int boot_device,

                            DisplayState *ds, const char **fd_filename, int snapshot,

                            const char *kernel_filename, const char *kernel_cmdline,

                            const char *initrd_filename, const char *cpu_model)

{

    if (cpu_model == NULL)

        cpu_model = "TI SuperSparc II";

    sun4m_common_init(ram_size, boot_device, ds, kernel_filename,

                      kernel_cmdline, initrd_filename, cpu_model,

                      1, PROM_ADDR); // XXX prom overlap, actually first 4GB ok

}
