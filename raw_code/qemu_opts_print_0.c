int qemu_opts_print(QemuOpts *opts, void *dummy)

{

    QemuOpt *opt;



    fprintf(stderr, "%s: %s:", opts->list->name,

            opts->id ? opts->id : "<noid>");

    TAILQ_FOREACH(opt, &opts->head, next) {

        fprintf(stderr, " %s=\"%s\"", opt->name, opt->str);

    }

    fprintf(stderr, "\n");

    return 0;

}
