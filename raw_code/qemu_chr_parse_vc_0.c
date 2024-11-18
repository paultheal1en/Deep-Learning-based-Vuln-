static void qemu_chr_parse_vc(QemuOpts *opts, ChardevBackend *backend,

                              Error **errp)

{

    int val;



    backend->vc = g_new0(ChardevVC, 1);



    val = qemu_opt_get_number(opts, "width", 0);

    if (val != 0) {

        backend->vc->has_width = true;

        backend->vc->width = val;

    }



    val = qemu_opt_get_number(opts, "height", 0);

    if (val != 0) {

        backend->vc->has_height = true;

        backend->vc->height = val;

    }



    val = qemu_opt_get_number(opts, "cols", 0);

    if (val != 0) {

        backend->vc->has_cols = true;

        backend->vc->cols = val;

    }



    val = qemu_opt_get_number(opts, "rows", 0);

    if (val != 0) {

        backend->vc->has_rows = true;

        backend->vc->rows = val;

    }

}
