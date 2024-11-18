int vnc_display_pw_expire(DisplayState *ds, time_t expires)
{
    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;
    vs->expires = expires;
    return 0;