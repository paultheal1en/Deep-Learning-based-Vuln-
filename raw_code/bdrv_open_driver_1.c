static int bdrv_open_driver(BlockDriverState *bs, BlockDriver *drv,

                            const char *node_name, QDict *options,

                            int open_flags, Error **errp)

{

    Error *local_err = NULL;

    int ret;



    bdrv_assign_node_name(bs, node_name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return -EINVAL;

    }



    bs->drv = drv;

    bs->read_only = !(bs->open_flags & BDRV_O_RDWR);

    bs->opaque = g_malloc0(drv->instance_size);



    if (drv->bdrv_file_open) {

        assert(!drv->bdrv_needs_filename || bs->filename[0]);

        ret = drv->bdrv_file_open(bs, options, open_flags, &local_err);

    } else if (drv->bdrv_open) {

        ret = drv->bdrv_open(bs, options, open_flags, &local_err);

    } else {

        ret = 0;

    }



    if (ret < 0) {

        if (local_err) {

            error_propagate(errp, local_err);

        } else if (bs->filename[0]) {

            error_setg_errno(errp, -ret, "Could not open '%s'", bs->filename);

        } else {

            error_setg_errno(errp, -ret, "Could not open image");

        }

        goto free_and_fail;

    }



    ret = refresh_total_sectors(bs, bs->total_sectors);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not refresh total sector count");

        goto free_and_fail;

    }



    bdrv_refresh_limits(bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto free_and_fail;

    }



    assert(bdrv_opt_mem_align(bs) != 0);

    assert(bdrv_min_mem_align(bs) != 0);

    assert(is_power_of_2(bs->bl.request_alignment));



    return 0;



free_and_fail:

    /* FIXME Close bs first if already opened*/

    g_free(bs->opaque);

    bs->opaque = NULL;

    bs->drv = NULL;

    return ret;

}
