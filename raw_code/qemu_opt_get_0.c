const char *qemu_opt_get(QemuOpts *opts, const char *name)

{

    QemuOpt *opt = qemu_opt_find(opts, name);



    if (!opt) {

        const QemuOptDesc *desc = find_desc_by_name(opts->list->desc, name);

        if (desc && desc->def_value_str) {

            return desc->def_value_str;

        }

    }

    return opt ? opt->str : NULL;

}
