static const char *bdrv_get_parent_name(const BlockDriverState *bs)

{

    BdrvChild *c;

    const char *name;



    /* If multiple parents have a name, just pick the first one. */

    QLIST_FOREACH(c, &bs->parents, next_parent) {

        if (c->role->get_name) {

            name = c->role->get_name(c);

            if (name && *name) {

                return name;

            }

        }

    }



    return NULL;

}
