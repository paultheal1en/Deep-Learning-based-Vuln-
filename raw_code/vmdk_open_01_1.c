static int vmdk_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    char *buf;

    int ret;

    BDRVVmdkState *s = bs->opaque;

    uint32_t magic;

    Error *local_err = NULL;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    buf = vmdk_read_desc(bs->file, 0, errp);

    if (!buf) {

        return -EINVAL;

    }



    magic = ldl_be_p(buf);

    switch (magic) {

        case VMDK3_MAGIC:

        case VMDK4_MAGIC:

            ret = vmdk_open_sparse(bs, bs->file, flags, buf, options,

                                   errp);

            s->desc_offset = 0x200;

            break;

        default:

            ret = vmdk_open_desc_file(bs, flags, buf, options, errp);

            break;

    }

    if (ret) {

        goto fail;

    }



    /* try to open parent images, if exist */

    ret = vmdk_parent_open(bs);

    if (ret) {

        goto fail;

    }

    s->cid = vmdk_read_cid(bs, 0);

    s->parent_cid = vmdk_read_cid(bs, 1);

    qemu_co_mutex_init(&s->lock);



    /* Disable migration when VMDK images are used */

    error_setg(&s->migration_blocker, "The vmdk format used by node '%s' "

               "does not support live migration",

               bdrv_get_device_or_node_name(bs));

    ret = migrate_add_blocker(s->migration_blocker, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        error_free(s->migration_blocker);

        goto fail;

    }



    g_free(buf);

    return 0;



fail:

    g_free(buf);

    g_free(s->create_type);

    s->create_type = NULL;

    vmdk_free_extents(bs);

    return ret;

}
