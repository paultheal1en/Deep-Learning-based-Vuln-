static int xenfb_send_motion(struct XenInput *xenfb,

			     int rel_x, int rel_y, int rel_z)

{

    union xenkbd_in_event event;



    memset(&event, 0, XENKBD_IN_EVENT_SIZE);

    event.type = XENKBD_TYPE_MOTION;

    event.motion.rel_x = rel_x;

    event.motion.rel_y = rel_y;

#if __XEN_LATEST_INTERFACE_VERSION__ >= 0x00030207

    event.motion.rel_z = rel_z;

#endif



    return xenfb_kbd_event(xenfb, &event);

}
