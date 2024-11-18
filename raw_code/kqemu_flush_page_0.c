void kqemu_flush_page(CPUState *env, target_ulong addr)

{

    LOG_INT("kqemu_flush_page: addr=" TARGET_FMT_lx "\n", addr);

    if (nb_pages_to_flush >= KQEMU_MAX_PAGES_TO_FLUSH)

        nb_pages_to_flush = KQEMU_FLUSH_ALL;

    else

        pages_to_flush[nb_pages_to_flush++] = addr;

}
