static void ppce500_reset_device_tree(void *opaque)

{

    DeviceTreeParams *p = opaque;

    ppce500_load_device_tree(p->machine, &p->params, p->addr, p->initrd_base,

                             p->initrd_size, false);

}
