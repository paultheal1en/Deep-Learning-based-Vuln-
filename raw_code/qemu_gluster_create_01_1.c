static int qemu_gluster_create(const char *filename,

        QEMUOptionParameter *options, Error **errp)

{

    struct glfs *glfs;

    struct glfs_fd *fd;

    int ret = 0;

    int prealloc = 0;

    int64_t total_size = 0;

    GlusterConf *gconf = g_malloc0(sizeof(GlusterConf));



    glfs = qemu_gluster_init(gconf, filename, errp);

    if (!glfs) {

        ret = -EINVAL;

        goto out;

    }



    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            total_size = options->value.n / BDRV_SECTOR_SIZE;

        } else if (!strcmp(options->name, BLOCK_OPT_PREALLOC)) {

            if (!options->value.s || !strcmp(options->value.s, "off")) {

                prealloc = 0;

            } else if (!strcmp(options->value.s, "full") &&

                    gluster_supports_zerofill()) {

                prealloc = 1;

            } else {

                error_setg(errp, "Invalid preallocation mode: '%s'"

                    " or GlusterFS doesn't support zerofill API",

                           options->value.s);

                ret = -EINVAL;

                goto out;

            }

        }

        options++;

    }



    fd = glfs_creat(glfs, gconf->image,

        O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, S_IRUSR | S_IWUSR);

    if (!fd) {

        ret = -errno;

    } else {

        if (!glfs_ftruncate(fd, total_size * BDRV_SECTOR_SIZE)) {

            if (prealloc && qemu_gluster_zerofill(fd, 0,

                    total_size * BDRV_SECTOR_SIZE)) {

                ret = -errno;

            }

        } else {

            ret = -errno;

        }



        if (glfs_close(fd) != 0) {

            ret = -errno;

        }

    }

out:

    qemu_gluster_gconf_free(gconf);

    if (glfs) {

        glfs_fini(glfs);

    }

    return ret;

}
