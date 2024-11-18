static void get_xbzrle_cache_stats(MigrationInfo *info)

{

    if (migrate_use_xbzrle()) {

        info->has_xbzrle_cache = true;

        info->xbzrle_cache = g_malloc0(sizeof(*info->xbzrle_cache));

        info->xbzrle_cache->cache_size = migrate_xbzrle_cache_size();

        info->xbzrle_cache->bytes = xbzrle_mig_bytes_transferred();

        info->xbzrle_cache->pages = xbzrle_mig_pages_transferred();

        info->xbzrle_cache->cache_miss = xbzrle_mig_pages_cache_miss();

        info->xbzrle_cache->cache_miss_rate = xbzrle_mig_cache_miss_rate();

        info->xbzrle_cache->overflow = xbzrle_mig_pages_overflow();

    }

}
