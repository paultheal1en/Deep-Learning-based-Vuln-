void bdrv_parent_drained_end(BlockDriverState *bs)

{

    BdrvChild *c;



    QLIST_FOREACH(c, &bs->parents, next_parent) {

        if (c->role->drained_end) {

            c->role->drained_end(c);

        }

    }

}
