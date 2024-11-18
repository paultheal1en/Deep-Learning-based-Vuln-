static int input_initialise(struct XenDevice *xendev)

{

    struct XenInput *in = container_of(xendev, struct XenInput, c.xendev);

    int rc;



    if (!in->c.con) {

        xen_pv_printf(xendev, 1, "ds not set (yet)\n");

        return -1;

    }



    rc = common_bind(&in->c);

    if (rc != 0)

	return rc;



    qemu_add_kbd_event_handler(xenfb_key_event, in);

    return 0;

}
