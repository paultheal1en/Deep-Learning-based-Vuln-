void cache_insert(PageCache *cache, uint64_t addr, uint8_t *pdata)
{
    CacheItem *it = NULL;
    g_assert(cache);
    g_assert(cache->page_cache);
    /* actual update of entry */
    it = cache_get_by_addr(cache, addr);
    if (!it->it_data) {
        cache->num_items++;
    }
    it->it_data = pdata;
    it->it_age = ++cache->max_item_age;
    it->it_addr = addr;
}