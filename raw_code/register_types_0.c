static void register_types(void)

{

    register_char_driver_qapi("null", CHARDEV_BACKEND_KIND_NULL, NULL);

    register_char_driver("socket", qemu_chr_open_socket);

    register_char_driver("udp", qemu_chr_open_udp);

    register_char_driver("memory", qemu_chr_open_ringbuf);

    register_char_driver_qapi("file", CHARDEV_BACKEND_KIND_FILE,

                              qemu_chr_parse_file_out);

    register_char_driver_qapi("stdio", CHARDEV_BACKEND_KIND_STDIO,

                              qemu_chr_parse_stdio);

    register_char_driver_qapi("serial", CHARDEV_BACKEND_KIND_SERIAL,

                              qemu_chr_parse_serial);

    register_char_driver_qapi("tty", CHARDEV_BACKEND_KIND_SERIAL,

                              qemu_chr_parse_serial);

    register_char_driver_qapi("parallel", CHARDEV_BACKEND_KIND_PARALLEL,

                              qemu_chr_parse_parallel);

    register_char_driver_qapi("parport", CHARDEV_BACKEND_KIND_PARALLEL,

                              qemu_chr_parse_parallel);

#ifdef _WIN32

    register_char_driver("pipe", qemu_chr_open_win_pipe);

    register_char_driver("console", qemu_chr_open_win_con);

#else

    register_char_driver("pipe", qemu_chr_open_pipe);

#endif

#ifdef HAVE_CHARDEV_TTY

    register_char_driver("pty", qemu_chr_open_pty);

#endif

}
