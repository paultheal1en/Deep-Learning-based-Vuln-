ChardevReturn *qmp_chardev_add(const char *id, ChardevBackend *backend,

                               Error **errp)

{

    ChardevReturn *ret = g_new0(ChardevReturn, 1);

    CharDriverState *base, *chr = NULL;



    chr = qemu_chr_find(id);

    if (chr) {

        error_setg(errp, "Chardev '%s' already exists", id);

        g_free(ret);

        return NULL;

    }



    switch (backend->kind) {

    case CHARDEV_BACKEND_KIND_FILE:

        chr = qmp_chardev_open_file(backend->file, errp);

        break;

    case CHARDEV_BACKEND_KIND_SERIAL:

        chr = qmp_chardev_open_serial(backend->serial, errp);

        break;

    case CHARDEV_BACKEND_KIND_PARALLEL:

        chr = qmp_chardev_open_parallel(backend->parallel, errp);

        break;

    case CHARDEV_BACKEND_KIND_SOCKET:

        chr = qmp_chardev_open_socket(backend->socket, errp);

        break;

#ifdef HAVE_CHARDEV_TTY

    case CHARDEV_BACKEND_KIND_PTY:

    {

        /* qemu_chr_open_pty sets "path" in opts */

        QemuOpts *opts;

        opts = qemu_opts_create_nofail(qemu_find_opts("chardev"));

        chr = qemu_chr_open_pty(opts);

        ret->pty = g_strdup(qemu_opt_get(opts, "path"));

        ret->has_pty = true;

        qemu_opts_del(opts);

        break;

    }

#endif

    case CHARDEV_BACKEND_KIND_NULL:

        chr = qemu_chr_open_null();

        break;

    case CHARDEV_BACKEND_KIND_MUX:

        base = qemu_chr_find(backend->mux->chardev);

        if (base == NULL) {

            error_setg(errp, "mux: base chardev %s not found",

                       backend->mux->chardev);

            break;

        }

        chr = qemu_chr_open_mux(base);

        break;

    case CHARDEV_BACKEND_KIND_MSMOUSE:

        chr = qemu_chr_open_msmouse();

        break;

#ifdef CONFIG_BRLAPI

    case CHARDEV_BACKEND_KIND_BRAILLE:

        chr = chr_baum_init();

        break;

#endif

    case CHARDEV_BACKEND_KIND_STDIO:

        chr = qemu_chr_open_stdio(backend->stdio);

        break;

    default:

        error_setg(errp, "unknown chardev backend (%d)", backend->kind);

        break;

    }



    if (chr == NULL && !error_is_set(errp)) {

        error_setg(errp, "Failed to create chardev");

    }

    if (chr) {

        chr->label = g_strdup(id);

        chr->avail_connections =

            (backend->kind == CHARDEV_BACKEND_KIND_MUX) ? MAX_MUX : 1;

        QTAILQ_INSERT_TAIL(&chardevs, chr, next);

        return ret;

    } else {

        g_free(ret);

        return NULL;

    }

}
