int qemu_opt_set_bool(QemuOpts *opts, const char *name, bool val)

{

    QemuOpt *opt;

    const QemuOptDesc *desc = opts->list->desc;

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

            qerror_report(QERR_INVALID_PARAMETER, name);

            return -1;

        }

    }



    opt = g_malloc0(sizeof(*opt));

    opt->name = g_strdup(name);

    opt->opts = opts;

    QTAILQ_INSERT_TAIL(&opts->head, opt, next);

    if (desc[i].name != NULL) {

        opt->desc = desc+i;

    }

    opt->value.boolean = !!val;

    return 0;

}
