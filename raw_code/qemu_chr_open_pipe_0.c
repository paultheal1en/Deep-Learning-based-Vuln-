static CharDriverState *qemu_chr_open_pipe(const char *id,

                                           ChardevBackend *backend,

                                           ChardevReturn *ret,

                                           Error **errp)

{

    ChardevHostdev *opts = backend->u.pipe;

    const char *filename = opts->device;

    CharDriverState *chr;

    WinCharState *s;



    chr = qemu_chr_alloc();

    s = g_new0(WinCharState, 1);

    chr->opaque = s;

    chr->chr_write = win_chr_write;

    chr->chr_close = win_chr_close;



    if (win_chr_pipe_init(chr, filename, errp) < 0) {

        g_free(s);

        g_free(chr);

        return NULL;

    }

    return chr;

}
