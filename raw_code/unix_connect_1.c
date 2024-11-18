int unix_connect(const char *path)

{

    QemuOpts *opts;

    int sock;



    opts = qemu_opts_create(&dummy_opts, NULL, 0);

    qemu_opt_set(opts, "path", path);

    sock = unix_connect_opts(opts);

    qemu_opts_del(opts);

    return sock;

}
