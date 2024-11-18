void vnc_flush(VncState *vs)

{

    vnc_lock_output(vs);

    if (vs->csock != -1 && (vs->output.offset

#ifdef CONFIG_VNC_WS

                || vs->ws_output.offset

#endif

                )) {

        vnc_client_write_locked(vs);

    }

    vnc_unlock_output(vs);

}
