static void ss5_init(int ram_size, int vga_ram_size, int boot_device,

                       DisplayState *ds, const char **fd_filename, int snapshot,

                       const char *kernel_filename, const char *kernel_cmdline,

                       const char *initrd_filename, const char *cpu_model)

{

    if (cpu_model == NULL)

        cpu_model = "Fujitsu MB86904";

    sun4m_common_init(ram_size, boot_device, ds, kernel_filename,

                      kernel_cmdline, initrd_filename, cpu_model,

                      0);

}
