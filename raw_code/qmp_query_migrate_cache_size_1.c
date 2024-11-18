int64_t qmp_query_migrate_cache_size(Error **errp)

{

    return migrate_xbzrle_cache_size();

}
