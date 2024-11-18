static BlockBackend *img_open_file(const char *filename,

                                   QDict *options,

                                   const char *fmt, int flags,

                                   bool writethrough, bool quiet,

                                   bool force_share)

{

    BlockBackend *blk;

    Error *local_err = NULL;



    if (!options) {

        options = qdict_new();

    }

    if (fmt) {

        qdict_put_str(options, "driver", fmt);

    }



    if (force_share) {

        qdict_put_bool(options, BDRV_OPT_FORCE_SHARE, true);

    }

    blk = blk_new_open(filename, NULL, options, flags, &local_err);

    if (!blk) {

        error_reportf_err(local_err, "Could not open '%s': ", filename);

        return NULL;

    }

    blk_set_enable_write_cache(blk, !writethrough);



    if (img_open_password(blk, filename, flags, quiet) < 0) {

        blk_unref(blk);

        return NULL;

    }

    return blk;

}
