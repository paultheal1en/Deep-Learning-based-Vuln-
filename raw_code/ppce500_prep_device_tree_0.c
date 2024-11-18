static int ppce500_prep_device_tree(MachineState *machine,

                                    PPCE500Params *params,

                                    hwaddr addr,

                                    hwaddr initrd_base,

                                    hwaddr initrd_size)

{

    DeviceTreeParams *p = g_new(DeviceTreeParams, 1);

    p->machine = machine;

    p->params = *params;

    p->addr = addr;

    p->initrd_base = initrd_base;

    p->initrd_size = initrd_size;



    qemu_register_reset(ppce500_reset_device_tree, p);



    /* Issue the device tree loader once, so that we get the size of the blob */

    return ppce500_load_device_tree(machine, params, addr, initrd_base,

                                    initrd_size, true);

}
