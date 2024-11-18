QemuOpts *qemu_opts_create(QemuOptsList *list, const char *id, int fail_if_exists)

{

    QemuOpts *opts = NULL;



    if (id) {

        opts = qemu_opts_find(list, id);

        if (opts != NULL) {

            if (fail_if_exists) {

                fprintf(stderr, "tried to create id \"%s\" twice for \"%s\"\n",

                        id, list->name);

                return NULL;

            } else {

                return opts;

            }

        }

    }

    opts = qemu_mallocz(sizeof(*opts));

    if (id) {

        opts->id = qemu_strdup(id);

    }

    opts->list = list;

    TAILQ_INIT(&opts->head);

    TAILQ_INSERT_TAIL(&list->head, opts, next);

    return opts;

}
