static bool append_open_options(QDict *d, BlockDriverState *bs)

{

    const QDictEntry *entry;

    QemuOptDesc *desc;

    bool found_any = false;



    for (entry = qdict_first(bs->options); entry;

         entry = qdict_next(bs->options, entry))

    {

        /* Only take options for this level */

        if (strchr(qdict_entry_key(entry), '.')) {

            continue;

        }



        /* And exclude all non-driver-specific options */

        for (desc = bdrv_runtime_opts.desc; desc->name; desc++) {

            if (!strcmp(qdict_entry_key(entry), desc->name)) {

                break;

            }

        }

        if (desc->name) {

            continue;

        }



        qobject_incref(qdict_entry_value(entry));

        qdict_put_obj(d, qdict_entry_key(entry), qdict_entry_value(entry));

        found_any = true;

    }



    return found_any;

}
