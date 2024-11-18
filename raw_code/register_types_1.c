static void register_types(void)

{

    register_char_driver("null", CHARDEV_BACKEND_KIND_NULL, NULL,

                         qemu_chr_open_null);

    register_char_driver("socket", CHARDEV_BACKEND_KIND_SOCKET,

                         qemu_chr_parse_socket, qmp_chardev_open_socket);

    register_char_driver("udp", CHARDEV_BACKEND_KIND_UDP, qemu_chr_parse_udp,

                         qmp_chardev_open_udp);

    register_char_driver("ringbuf", CHARDEV_BACKEND_KIND_RINGBUF,

                         qemu_chr_parse_ringbuf, qemu_chr_open_ringbuf);

    register_char_driver("file", CHARDEV_BACKEND_KIND_FILE,

                         qemu_chr_parse_file_out, qmp_chardev_open_file);

    register_char_driver("stdio", CHARDEV_BACKEND_KIND_STDIO,

                         qemu_chr_parse_stdio, qemu_chr_open_stdio);

#if defined HAVE_CHARDEV_SERIAL

    register_char_driver("serial", CHARDEV_BACKEND_KIND_SERIAL,

                         qemu_chr_parse_serial, qmp_chardev_open_serial);

    register_char_driver("tty", CHARDEV_BACKEND_KIND_SERIAL,

                         qemu_chr_parse_serial, qmp_chardev_open_serial);

#endif

#ifdef HAVE_CHARDEV_PARPORT

    register_char_driver("parallel", CHARDEV_BACKEND_KIND_PARALLEL,

                         qemu_chr_parse_parallel, qmp_chardev_open_parallel);

    register_char_driver("parport", CHARDEV_BACKEND_KIND_PARALLEL,

                         qemu_chr_parse_parallel, qmp_chardev_open_parallel);

#endif

#ifdef HAVE_CHARDEV_PTY

    register_char_driver("pty", CHARDEV_BACKEND_KIND_PTY, NULL,

                         qemu_chr_open_pty);

#endif

#ifdef _WIN32

    register_char_driver("console", CHARDEV_BACKEND_KIND_CONSOLE, NULL,

                         qemu_chr_open_win_con);

#endif

    register_char_driver("pipe", CHARDEV_BACKEND_KIND_PIPE,

                         qemu_chr_parse_pipe, qemu_chr_open_pipe);

    register_char_driver("mux", CHARDEV_BACKEND_KIND_MUX, qemu_chr_parse_mux,

                         qemu_chr_open_mux);

    /* Bug-compatibility: */

    register_char_driver("memory", CHARDEV_BACKEND_KIND_MEMORY,

                         qemu_chr_parse_ringbuf, qemu_chr_open_ringbuf);

    /* this must be done after machine init, since we register FEs with muxes

     * as part of realize functions like serial_isa_realizefn when -nographic

     * is specified

     */

    qemu_add_machine_init_done_notifier(&muxes_realize_notify);



    atexit(qemu_chr_cleanup);

}
