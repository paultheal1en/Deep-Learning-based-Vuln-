static void glib_pollfds_poll(void)

{

    GMainContext *context = g_main_context_default();

    GPollFD *pfds = &g_array_index(gpollfds, GPollFD, glib_pollfds_idx);



    if (g_main_context_check(context, max_priority, pfds, glib_n_poll_fds)) {

        g_main_context_dispatch(context);

    }

}
