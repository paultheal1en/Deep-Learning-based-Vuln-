static bool object_create_initial(const char *type)

{

    if (g_str_equal(type, "rng-egd")) {

        return false;

    }



    /*

     * return false for concrete netfilters since

     * they depend on netdevs already existing

     */

    if (g_str_equal(type, "filter-buffer") ||

        g_str_equal(type, "filter-dump") ||

        g_str_equal(type, "filter-mirror") ||

        g_str_equal(type, "filter-redirector") ||

        g_str_equal(type, "colo-compare") ||

        g_str_equal(type, "filter-rewriter")) {

        return false;

    }



    /* Memory allocation by backends needs to be done

     * after configure_accelerator() (due to the tcg_enabled()

     * checks at memory_region_init_*()).

     *

     * Also, allocation of large amounts of memory may delay

     * chardev initialization for too long, and trigger timeouts

     * on software that waits for a monitor socket to be created

     * (e.g. libvirt).

     */

    if (g_str_has_prefix(type, "memory-backend-")) {

        return false;

    }



    return true;

}
