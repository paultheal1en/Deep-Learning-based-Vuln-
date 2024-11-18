int bdrv_open_image(BlockDriverState **pbs, const char *filename,

                    QDict *options, const char *bdref_key, int flags,

                    bool allow_none, Error **errp)

{

    QDict *image_options;

    int ret;

    char *bdref_key_dot;

    const char *reference;



    assert(pbs);

    assert(*pbs == NULL);



    bdref_key_dot = g_strdup_printf("%s.", bdref_key);

    qdict_extract_subqdict(options, &image_options, bdref_key_dot);

    g_free(bdref_key_dot);



    reference = qdict_get_try_str(options, bdref_key);

    if (!filename && !reference && !qdict_size(image_options)) {

        if (allow_none) {

            ret = 0;

        } else {

            error_setg(errp, "A block device must be specified for \"%s\"",

                       bdref_key);

            ret = -EINVAL;

        }


        goto done;

    }



    ret = bdrv_open(pbs, filename, reference, image_options, flags, NULL, errp);



done:

    qdict_del(options, bdref_key);

    return ret;

}