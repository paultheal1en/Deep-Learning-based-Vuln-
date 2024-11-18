BdrvChild *bdrv_root_attach_child(BlockDriverState *child_bs,

                                  const char *child_name,

                                  const BdrvChildRole *child_role,

                                  void *opaque)

{

    BdrvChild *child = g_new(BdrvChild, 1);

    *child = (BdrvChild) {

        .bs     = NULL,

        .name   = g_strdup(child_name),

        .role   = child_role,

        .opaque = opaque,

    };



    bdrv_replace_child(child, child_bs);



    return child;

}
