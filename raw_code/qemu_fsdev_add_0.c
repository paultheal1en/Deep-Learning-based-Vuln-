int qemu_fsdev_add(QemuOpts *opts)

{

    struct FsTypeListEntry *fsle;

    int i;



    if (qemu_opts_id(opts) == NULL) {

        fprintf(stderr, "fsdev: No id specified\n");

        return -1;

    }



     for (i = 0; i < ARRAY_SIZE(FsTypes); i++) {

        if (strcmp(FsTypes[i].name, qemu_opt_get(opts, "fstype")) == 0) {

            break;

        }

    }



    if (i == ARRAY_SIZE(FsTypes)) {

        fprintf(stderr, "fsdev: fstype %s not found\n",

                    qemu_opt_get(opts, "fstype"));

        return -1;

    }



    fsle = qemu_malloc(sizeof(*fsle));



    fsle->fse.fsdev_id = qemu_strdup(qemu_opts_id(opts));

    fsle->fse.path = qemu_strdup(qemu_opt_get(opts, "path"));

    fsle->fse.ops = FsTypes[i].ops;



    QTAILQ_INSERT_TAIL(&fstype_entries, fsle, next);

    return 0;



}
