static void do_info_trace(Monitor *mon)

{

    st_print_trace((FILE *)mon, &monitor_fprintf);

}
