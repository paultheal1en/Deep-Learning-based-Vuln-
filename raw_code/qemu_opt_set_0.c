int qemu_opt_set(QemuOpts *opts, const char *name, const char *value)

{

    QemuOpt *opt;



    opt = qemu_opt_find(opts, name);

    if (!opt) {

        QemuOptDesc *desc = opts->list->desc;

        int i;



        for (i = 0; desc[i].name != NULL; i++) {

            if (strcmp(desc[i].name, name) == 0) {

                break;

            }

        }

        if (desc[i].name == NULL) {

            if (i == 0) {

                /* empty list -> allow any */;

            } else {

                fprintf(stderr, "option \"%s\" is not valid for %s\n",

                        name, opts->list->name);

                return -1;

            }

        }

        opt = qemu_mallocz(sizeof(*opt));

        opt->name = qemu_strdup(name);

        opt->opts = opts;

        TAILQ_INSERT_TAIL(&opts->head, opt, next);

        if (desc[i].name != NULL) {

            opt->desc = desc+i;

        }

    }

    qemu_free((/* !const */ char*)opt->str);

    opt->str = NULL;

    if (value) {

        opt->str = qemu_strdup(value);

    }

    if (qemu_opt_parse(opt) < 0) {

        fprintf(stderr, "Failed to parse \"%s\" for \"%s.%s\"\n", opt->str,

                opts->list->name, opt->name);

        qemu_opt_del(opt);

        return -1;

    }

    return 0;

}
