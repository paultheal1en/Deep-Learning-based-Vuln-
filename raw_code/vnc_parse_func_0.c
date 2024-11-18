QemuOpts *vnc_parse_func(const char *str)

{

    QemuOptsList *olist = qemu_find_opts("vnc");

    QemuOpts *opts = qemu_opts_parse(olist, str, 1);

    const char *id = qemu_opts_id(opts);



    if (!id) {

        /* auto-assign id if not present */

        vnc_auto_assign_id(olist, opts);

    }

    return opts;

}
