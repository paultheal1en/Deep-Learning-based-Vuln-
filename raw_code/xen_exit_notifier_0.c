static void xen_exit_notifier(Notifier *n, void *data)

{

    XenIOState *state = container_of(n, XenIOState, exit);



    xc_evtchn_close(state->xce_handle);

    xs_daemon_close(state->xenstore);

}
