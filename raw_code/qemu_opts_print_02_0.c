void qemu_opts_print(QemuOpts *opts)

{

    QemuOpt *opt;

    QemuOptDesc *desc = opts->list->desc;



    if (desc[0].name == NULL) {

        QTAILQ_FOREACH(opt, &opts->head, next) {

            printf("%s=\"%s\" ", opt->name, opt->str);

        }

        return;

    }

    for (; desc && desc->name; desc++) {

        const char *value;

        QemuOpt *opt = qemu_opt_find(opts, desc->name);



        value = opt ? opt->str : desc->def_value_str;

        if (!value) {

            continue;

        }

        if (desc->type == QEMU_OPT_STRING) {

            printf("%s='%s' ", desc->name, value);

        } else if ((desc->type == QEMU_OPT_SIZE ||

                    desc->type == QEMU_OPT_NUMBER) && opt) {

            printf("%s=%" PRId64 " ", desc->name, opt->value.uint);

        } else {

            printf("%s=%s ", desc->name, value);

        }

    }

}
