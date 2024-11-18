int qemu_global_option(const char *str)

{

    char driver[64], property[64];

    QemuOpts *opts;

    int rc, offset;



    rc = sscanf(str, "%63[^.].%63[^=]%n", driver, property, &offset);

    if (rc < 2 || str[offset] != '=') {

        error_report("can't parse: \"%s\"", str);

        return -1;

    }



    opts = qemu_opts_create(&qemu_global_opts, NULL, 0);

    qemu_opt_set(opts, "driver", driver);

    qemu_opt_set(opts, "property", property);

    qemu_opt_set(opts, "value", str+offset+1);

    return 0;

}
