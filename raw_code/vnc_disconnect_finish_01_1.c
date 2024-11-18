static void vnc_disconnect_finish(VncState *vs)
{
    vnc_qmp_event(vs, QEVENT_VNC_DISCONNECTED);
    buffer_free(&vs->input);
    buffer_free(&vs->output);
    qobject_decref(vs->info);
#ifdef CONFIG_VNC_TLS
    vnc_tls_client_cleanup(vs);
#endif /* CONFIG_VNC_TLS */
#ifdef CONFIG_VNC_SASL
    vnc_sasl_client_cleanup(vs);
#endif /* CONFIG_VNC_SASL */
    audio_del(vs);
    QTAILQ_REMOVE(&vs->vd->clients, vs, next);
    if (QTAILQ_EMPTY(&vs->vd->clients)) {
        dcl->idle = 1;
    }
    qemu_remove_mouse_mode_change_notifier(&vs->mouse_mode_notifier);
    vnc_remove_timer(vs->vd);
    if (vs->vd->lock_key_sync)
        qemu_remove_led_event_handler(vs->led);
    qemu_free(vs);
}