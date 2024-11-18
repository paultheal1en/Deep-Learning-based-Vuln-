CharDriverState *qemu_chr_new_from_opts(QemuOpts *opts,

                                    void (*init)(struct CharDriverState *s),

                                    Error **errp)

{

    CharDriver *cd;

    CharDriverState *chr;

    GSList *i;



    if (qemu_opts_id(opts) == NULL) {

        error_setg(errp, "chardev: no id specified");

        goto err;

    }



    if (qemu_opt_get(opts, "backend") == NULL) {

        error_setg(errp, "chardev: \"%s\" missing backend",

                   qemu_opts_id(opts));

        goto err;

    }

    for (i = backends; i; i = i->next) {

        cd = i->data;



        if (strcmp(cd->name, qemu_opt_get(opts, "backend")) == 0) {

            break;

        }

    }

    if (i == NULL) {

        error_setg(errp, "chardev: backend \"%s\" not found",

                   qemu_opt_get(opts, "backend"));

        goto err;

    }



    if (!cd->open) {

        /* using new, qapi init */

        ChardevBackend *backend = g_new0(ChardevBackend, 1);

        ChardevReturn *ret = NULL;

        const char *id = qemu_opts_id(opts);

        const char *bid = NULL;



        if (qemu_opt_get_bool(opts, "mux", 0)) {

            bid = g_strdup_printf("%s-base", id);

        }



        chr = NULL;

        backend->kind = cd->kind;

        if (cd->parse) {

            cd->parse(opts, backend, errp);

            if (error_is_set(errp)) {

                goto qapi_out;

            }

        }

        ret = qmp_chardev_add(bid ? bid : id, backend, errp);

        if (error_is_set(errp)) {

            goto qapi_out;

        }



        if (bid) {

            qapi_free_ChardevBackend(backend);

            qapi_free_ChardevReturn(ret);

            backend = g_new0(ChardevBackend, 1);

            backend->mux = g_new0(ChardevMux, 1);

            backend->kind = CHARDEV_BACKEND_KIND_MUX;

            backend->mux->chardev = g_strdup(bid);

            ret = qmp_chardev_add(id, backend, errp);

            if (error_is_set(errp)) {

                goto qapi_out;

            }

        }



        chr = qemu_chr_find(id);



    qapi_out:

        qapi_free_ChardevBackend(backend);

        qapi_free_ChardevReturn(ret);

        return chr;

    }



    chr = cd->open(opts);

    if (!chr) {

        error_setg(errp, "chardev: opening backend \"%s\" failed",

                   qemu_opt_get(opts, "backend"));

        goto err;

    }



    if (!chr->filename)

        chr->filename = g_strdup(qemu_opt_get(opts, "backend"));

    chr->init = init;

    /* if we didn't create the chardev via qmp_chardev_add, we

     * need to send the OPENED event here

     */

    if (!chr->explicit_be_open) {

        qemu_chr_be_event(chr, CHR_EVENT_OPENED);

    }

    QTAILQ_INSERT_TAIL(&chardevs, chr, next);



    if (qemu_opt_get_bool(opts, "mux", 0)) {

        CharDriverState *base = chr;

        int len = strlen(qemu_opts_id(opts)) + 6;

        base->label = g_malloc(len);

        snprintf(base->label, len, "%s-base", qemu_opts_id(opts));

        chr = qemu_chr_open_mux(base);

        chr->filename = base->filename;

        chr->avail_connections = MAX_MUX;

        QTAILQ_INSERT_TAIL(&chardevs, chr, next);

    } else {

        chr->avail_connections = 1;

    }

    chr->label = g_strdup(qemu_opts_id(opts));

    chr->opts = opts;

    return chr;



err:

    qemu_opts_del(opts);

    return NULL;

}
