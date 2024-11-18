int64_t migrate_xbzrle_cache_size(void)

{

    MigrationState *s;



    s = migrate_get_current();



    return s->xbzrle_cache_size;

}
