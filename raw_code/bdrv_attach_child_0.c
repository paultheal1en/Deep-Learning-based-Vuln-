BdrvChild *bdrv_attach_child(BlockDriverState *parent_bs,

                             BlockDriverState *child_bs,

                             const char *child_name,

                             const BdrvChildRole *child_role,

                             Error **errp)

{

    BdrvChild *child = bdrv_root_attach_child(child_bs, child_name, child_role,

                                              parent_bs);

    QLIST_INSERT_HEAD(&parent_bs->children, child, next);

    return child;

}
