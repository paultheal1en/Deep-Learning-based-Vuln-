void qmp_x_blockdev_insert_medium(const char *device, const char *node_name,

                                  Error **errp)

{

    BlockDriverState *bs;



    bs = bdrv_find_node(node_name);

    if (!bs) {

        error_setg(errp, "Node '%s' not found", node_name);

        return;

    }



    if (bs->blk) {

        error_setg(errp, "Node '%s' is already in use by '%s'", node_name,

                   blk_name(bs->blk));

        return;

    }



    qmp_blockdev_insert_anon_medium(device, bs, errp);

}
