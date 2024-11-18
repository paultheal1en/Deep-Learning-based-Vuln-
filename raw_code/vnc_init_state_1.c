void vnc_init_state(VncState *vs)

{

    vs->initialized = true;

    VncDisplay *vd = vs->vd;



    vs->last_x = -1;

    vs->last_y = -1;



    vs->as.freq = 44100;

    vs->as.nchannels = 2;

    vs->as.fmt = AUD_FMT_S16;

    vs->as.endianness = 0;



    qemu_mutex_init(&vs->output_mutex);

    vs->bh = qemu_bh_new(vnc_jobs_bh, vs);



    QTAILQ_INSERT_HEAD(&vd->clients, vs, next);



    graphic_hw_update(vd->dcl.con);



    vnc_write(vs, "RFB 003.008\n", 12);

    vnc_flush(vs);

    vnc_read_when(vs, protocol_version, 12);

    reset_keys(vs);

    if (vs->vd->lock_key_sync)

        vs->led = qemu_add_led_event_handler(kbd_leds, vs);



    vs->mouse_mode_notifier.notify = check_pointer_type_change;

    qemu_add_mouse_mode_change_notifier(&vs->mouse_mode_notifier);



    /* vs might be free()ed here */

}
