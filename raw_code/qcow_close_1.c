static void qcow_close(BlockDriverState *bs)

{

    BDRVQcowState *s = bs->opaque;



    g_free(s->l1_table);

    g_free(s->l2_cache);

    g_free(s->cluster_cache);

    g_free(s->cluster_data);



    migrate_del_blocker(s->migration_blocker);

    error_free(s->migration_blocker);

}
