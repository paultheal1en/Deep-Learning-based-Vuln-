void qemu_opts_del(QemuOpts *opts)

{

    QemuOpt *opt;



    for (;;) {

        opt = TAILQ_FIRST(&opts->head);

        if (opt == NULL)

            break;

        qemu_opt_del(opt);

    }

    TAILQ_REMOVE(&opts->list->head, opts, next);

    qemu_free(opts);

}
