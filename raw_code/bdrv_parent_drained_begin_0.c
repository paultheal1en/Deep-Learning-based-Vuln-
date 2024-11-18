void bdrv_parent_drained_begin(BlockDriverState *bs)

{

    BdrvChild *c;



    QLIST_FOREACH(c, &bs->parents, next_parent) {

        if (c->role->drained_begin) {

            c->role->drained_begin(c);

        }

    }

}
