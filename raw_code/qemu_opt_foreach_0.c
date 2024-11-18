int qemu_opt_foreach(QemuOpts *opts, qemu_opt_loopfunc func, void *opaque,

                     int abort_on_failure)

{

    QemuOpt *opt;

    int rc = 0;



    TAILQ_FOREACH(opt, &opts->head, next) {

        rc = func(opt->name, opt->str, opaque);

        if (abort_on_failure  &&  rc != 0)

            break;

    }

    return rc;

}
