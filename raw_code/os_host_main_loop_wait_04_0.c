static int os_host_main_loop_wait(uint32_t timeout)

{

    int ret;



    glib_select_fill(&nfds, &rfds, &wfds, &xfds, &timeout);



    if (timeout > 0) {

        qemu_mutex_unlock_iothread();

    }



    /* We'll eventually drop fd_set completely.  But for now we still have

     * *_fill() and *_poll() functions that use rfds/wfds/xfds.

     */

    gpollfds_from_select();



    ret = g_poll((GPollFD *)gpollfds->data, gpollfds->len, timeout);



    gpollfds_to_select(ret);



    if (timeout > 0) {

        qemu_mutex_lock_iothread();

    }



    glib_select_poll(&rfds, &wfds, &xfds, (ret < 0));

    return ret;

}
