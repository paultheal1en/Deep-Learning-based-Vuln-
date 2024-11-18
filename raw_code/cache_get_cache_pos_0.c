static size_t cache_get_cache_pos(const PageCache *cache,

                                  uint64_t address)

{

    size_t pos;



    g_assert(cache->max_num_items);

    pos = (address / cache->page_size) & (cache->max_num_items - 1);

    return pos;

}
