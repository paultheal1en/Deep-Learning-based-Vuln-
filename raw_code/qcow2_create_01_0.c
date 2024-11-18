static int qcow2_create(const char *filename, QemuOpts *opts, Error **errp)

{

    char *backing_file = NULL;

    char *backing_fmt = NULL;

    char *buf = NULL;

    uint64_t size = 0;

    int flags = 0;

    size_t cluster_size = DEFAULT_CLUSTER_SIZE;

    PreallocMode prealloc;

    int version = 3;

    uint64_t refcount_bits = 16;

    int refcount_order;

    Error *local_err = NULL;

    int ret;



    /* Read out options */

    size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                    BDRV_SECTOR_SIZE);

    backing_file = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);

    backing_fmt = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FMT);

    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_ENCRYPT, false)) {

        flags |= BLOCK_FLAG_ENCRYPT;

    }

    cluster_size = qemu_opt_get_size_del(opts, BLOCK_OPT_CLUSTER_SIZE,

                                         DEFAULT_CLUSTER_SIZE);

    buf = qemu_opt_get_del(opts, BLOCK_OPT_PREALLOC);

    prealloc = qapi_enum_parse(PreallocMode_lookup, buf,

                               PREALLOC_MODE__MAX, PREALLOC_MODE_OFF,

                               &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto finish;

    }

    g_free(buf);

    buf = qemu_opt_get_del(opts, BLOCK_OPT_COMPAT_LEVEL);

    if (!buf) {

        /* keep the default */

    } else if (!strcmp(buf, "0.10")) {

        version = 2;

    } else if (!strcmp(buf, "1.1")) {

        version = 3;

    } else {

        error_setg(errp, "Invalid compatibility level: '%s'", buf);

        ret = -EINVAL;

        goto finish;

    }



    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_LAZY_REFCOUNTS, false)) {

        flags |= BLOCK_FLAG_LAZY_REFCOUNTS;

    }



    if (backing_file && prealloc != PREALLOC_MODE_OFF) {

        error_setg(errp, "Backing file and preallocation cannot be used at "

                   "the same time");

        ret = -EINVAL;

        goto finish;

    }



    if (version < 3 && (flags & BLOCK_FLAG_LAZY_REFCOUNTS)) {

        error_setg(errp, "Lazy refcounts only supported with compatibility "

                   "level 1.1 and above (use compat=1.1 or greater)");

        ret = -EINVAL;

        goto finish;

    }



    refcount_bits = qemu_opt_get_number_del(opts, BLOCK_OPT_REFCOUNT_BITS,

                                            refcount_bits);

    if (refcount_bits > 64 || !is_power_of_2(refcount_bits)) {

        error_setg(errp, "Refcount width must be a power of two and may not "

                   "exceed 64 bits");

        ret = -EINVAL;

        goto finish;

    }



    if (version < 3 && refcount_bits != 16) {

        error_setg(errp, "Different refcount widths than 16 bits require "

                   "compatibility level 1.1 or above (use compat=1.1 or "

                   "greater)");

        ret = -EINVAL;

        goto finish;

    }



    refcount_order = ctz32(refcount_bits);



    ret = qcow2_create2(filename, size, backing_file, backing_fmt, flags,

                        cluster_size, prealloc, opts, version, refcount_order,

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

    }



finish:

    g_free(backing_file);

    g_free(backing_fmt);

    g_free(buf);

    return ret;

}
