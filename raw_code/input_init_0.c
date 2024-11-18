static int input_init(struct XenDevice *xendev)

{

    struct XenInput *in = container_of(xendev, struct XenInput, c.xendev);



    if (!in->c.ds) {

        xen_be_printf(xendev, 1, "ds not set (yet)\n");

	return -1;

    }



    xenstore_write_be_int(xendev, "feature-abs-pointer", 1);

    return 0;

}
