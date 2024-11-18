static int raw_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int fd;

    int result = 0;

    int64_t total_size = 0;

    bool nocow = false;

    PreallocMode prealloc;

    char *buf = NULL;

    Error *local_err = NULL;



    strstart(filename, "file:", &filename);



    /* Read out options */

    total_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);

    nocow = qemu_opt_get_bool(opts, BLOCK_OPT_NOCOW, false);

    buf = qemu_opt_get_del(opts, BLOCK_OPT_PREALLOC);

    prealloc = qapi_enum_parse(PreallocMode_lookup, buf,

                               PREALLOC_MODE__MAX, PREALLOC_MODE_OFF,

                               &local_err);

    g_free(buf);

    if (local_err) {

        error_propagate(errp, local_err);

        result = -EINVAL;

        goto out;

    }



    fd = qemu_open(filename, O_RDWR | O_CREAT | O_TRUNC | O_BINARY,

                   0644);

    if (fd < 0) {

        result = -errno;

        error_setg_errno(errp, -result, "Could not create file");

        goto out;

    }



    if (nocow) {

#ifdef __linux__

        /* Set NOCOW flag to solve performance issue on fs like btrfs.

         * This is an optimisation. The FS_IOC_SETFLAGS ioctl return value

         * will be ignored since any failure of this operation should not

         * block the left work.

         */

        int attr;

        if (ioctl(fd, FS_IOC_GETFLAGS, &attr) == 0) {

            attr |= FS_NOCOW_FL;

            ioctl(fd, FS_IOC_SETFLAGS, &attr);

        }

#endif

    }



    if (ftruncate(fd, total_size) != 0) {

        result = -errno;

        error_setg_errno(errp, -result, "Could not resize file");

        goto out_close;

    }



    switch (prealloc) {

#ifdef CONFIG_POSIX_FALLOCATE

    case PREALLOC_MODE_FALLOC:

        /* posix_fallocate() doesn't set errno. */

        result = -posix_fallocate(fd, 0, total_size);

        if (result != 0) {

            error_setg_errno(errp, -result,

                             "Could not preallocate data for the new file");

        }

        break;

#endif

    case PREALLOC_MODE_FULL:

    {

        int64_t num = 0, left = total_size;

        buf = g_malloc0(65536);



        while (left > 0) {

            num = MIN(left, 65536);

            result = write(fd, buf, num);

            if (result < 0) {

                result = -errno;

                error_setg_errno(errp, -result,

                                 "Could not write to the new file");

                break;

            }

            left -= result;

        }

        if (result >= 0) {

            result = fsync(fd);

            if (result < 0) {

                result = -errno;

                error_setg_errno(errp, -result,

                                 "Could not flush new file to disk");

            }

        }

        g_free(buf);

        break;

    }

    case PREALLOC_MODE_OFF:

        break;

    default:

        result = -EINVAL;

        error_setg(errp, "Unsupported preallocation mode: %s",

                   PreallocMode_lookup[prealloc]);

        break;

    }



out_close:

    if (qemu_close(fd) != 0 && result == 0) {

        result = -errno;

        error_setg_errno(errp, -result, "Could not close the new file");

    }

out:

    return result;

}
