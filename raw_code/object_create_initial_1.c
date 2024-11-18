static bool object_create_initial(const char *type)
{
    if (g_str_equal(type, "rng-egd")) {
    /*
     * return false for concrete netfilters since
     * they depend on netdevs already existing
    if (g_str_equal(type, "filter-buffer") ||
        g_str_equal(type, "filter-dump") ||
        g_str_equal(type, "filter-mirror") ||
        g_str_equal(type, "filter-redirector")) {
    return true;