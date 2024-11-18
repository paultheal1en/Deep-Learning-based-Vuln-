static int qcow2_amend_options(BlockDriverState *bs, QemuOpts *opts,

                               BlockDriverAmendStatusCB *status_cb)

{

    BDRVQcowState *s = bs->opaque;

    int old_version = s->qcow_version, new_version = old_version;

    uint64_t new_size = 0;

    const char *backing_file = NULL, *backing_format = NULL;

    bool lazy_refcounts = s->use_lazy_refcounts;

    const char *compat = NULL;

    uint64_t cluster_size = s->cluster_size;

    bool encrypt;

    int ret;

    QemuOptDesc *desc = opts->list->desc;



    while (desc && desc->name) {

        if (!qemu_opt_find(opts, desc->name)) {

            /* only change explicitly defined options */

            desc++;

            continue;

        }



        if (!strcmp(desc->name, BLOCK_OPT_COMPAT_LEVEL)) {

            compat = qemu_opt_get(opts, BLOCK_OPT_COMPAT_LEVEL);

            if (!compat) {

                /* preserve default */

            } else if (!strcmp(compat, "0.10")) {

                new_version = 2;

            } else if (!strcmp(compat, "1.1")) {

                new_version = 3;

            } else {

                fprintf(stderr, "Unknown compatibility level %s.\n", compat);

                return -EINVAL;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_PREALLOC)) {

            fprintf(stderr, "Cannot change preallocation mode.\n");

            return -ENOTSUP;

        } else if (!strcmp(desc->name, BLOCK_OPT_SIZE)) {

            new_size = qemu_opt_get_size(opts, BLOCK_OPT_SIZE, 0);

        } else if (!strcmp(desc->name, BLOCK_OPT_BACKING_FILE)) {

            backing_file = qemu_opt_get(opts, BLOCK_OPT_BACKING_FILE);

        } else if (!strcmp(desc->name, BLOCK_OPT_BACKING_FMT)) {

            backing_format = qemu_opt_get(opts, BLOCK_OPT_BACKING_FMT);

        } else if (!strcmp(desc->name, BLOCK_OPT_ENCRYPT)) {

            encrypt = qemu_opt_get_bool(opts, BLOCK_OPT_ENCRYPT,

                                        s->crypt_method);

            if (encrypt != !!s->crypt_method) {

                fprintf(stderr, "Changing the encryption flag is not "

                        "supported.\n");

                return -ENOTSUP;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_CLUSTER_SIZE)) {

            cluster_size = qemu_opt_get_size(opts, BLOCK_OPT_CLUSTER_SIZE,

                                             cluster_size);

            if (cluster_size != s->cluster_size) {

                fprintf(stderr, "Changing the cluster size is not "

                        "supported.\n");

                return -ENOTSUP;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_LAZY_REFCOUNTS)) {

            lazy_refcounts = qemu_opt_get_bool(opts, BLOCK_OPT_LAZY_REFCOUNTS,

                                               lazy_refcounts);

        } else if (!strcmp(desc->name, BLOCK_OPT_REFCOUNT_BITS)) {

            error_report("Cannot change refcount entry width");

            return -ENOTSUP;

        } else {

            /* if this assertion fails, this probably means a new option was

             * added without having it covered here */

            assert(false);

        }



        desc++;

    }



    if (new_version != old_version) {

        if (new_version > old_version) {

            /* Upgrade */

            s->qcow_version = new_version;

            ret = qcow2_update_header(bs);

            if (ret < 0) {

                s->qcow_version = old_version;

                return ret;

            }

        } else {

            ret = qcow2_downgrade(bs, new_version, status_cb);

            if (ret < 0) {

                return ret;

            }

        }

    }



    if (backing_file || backing_format) {

        ret = qcow2_change_backing_file(bs, backing_file ?: bs->backing_file,

                                        backing_format ?: bs->backing_format);

        if (ret < 0) {

            return ret;

        }

    }



    if (s->use_lazy_refcounts != lazy_refcounts) {

        if (lazy_refcounts) {

            if (s->qcow_version < 3) {

                fprintf(stderr, "Lazy refcounts only supported with compatibility "

                        "level 1.1 and above (use compat=1.1 or greater)\n");

                return -EINVAL;

            }

            s->compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;

            ret = qcow2_update_header(bs);

            if (ret < 0) {

                s->compatible_features &= ~QCOW2_COMPAT_LAZY_REFCOUNTS;

                return ret;

            }

            s->use_lazy_refcounts = true;

        } else {

            /* make image clean first */

            ret = qcow2_mark_clean(bs);

            if (ret < 0) {

                return ret;

            }

            /* now disallow lazy refcounts */

            s->compatible_features &= ~QCOW2_COMPAT_LAZY_REFCOUNTS;

            ret = qcow2_update_header(bs);

            if (ret < 0) {

                s->compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;

                return ret;

            }

            s->use_lazy_refcounts = false;

        }

    }



    if (new_size) {

        ret = bdrv_truncate(bs, new_size);

        if (ret < 0) {

            return ret;

        }

    }



    return 0;

}
