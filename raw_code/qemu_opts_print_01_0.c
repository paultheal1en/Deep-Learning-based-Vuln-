void qemu_opts_print(QemuOpts *opts, const char *separator)

{

    QemuOpt *opt;

    QemuOptDesc *desc = opts->list->desc;

    const char *sep = "";



    if (opts->id) {

        printf("id=%s", opts->id); /* passed id_wellformed -> no commas */

        sep = separator;

    }



    if (desc[0].name == NULL) {

        QTAILQ_FOREACH(opt, &opts->head, next) {

            printf("%s%s=", sep, opt->name);

            escaped_print(opt->str);

            sep = separator;

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

            printf("%s%s=", sep, desc->name);

            escaped_print(value);

        } else if ((desc->type == QEMU_OPT_SIZE ||

                    desc->type == QEMU_OPT_NUMBER) && opt) {

            printf("%s%s=%" PRId64, sep, desc->name, opt->value.uint);

        } else {

            printf("%s%s=%s", sep, desc->name, value);

        }

        sep = separator;

    }

}
