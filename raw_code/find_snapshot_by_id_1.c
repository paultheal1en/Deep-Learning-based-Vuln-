static int find_snapshot_by_id(BlockDriverState *bs, const char *id_str)

{

    BDRVQcowState *s = bs->opaque;

    int i;



    for(i = 0; i < s->nb_snapshots; i++) {

        if (!strcmp(s->snapshots[i].id_str, id_str))

            return i;

    }

    return -1;

}
