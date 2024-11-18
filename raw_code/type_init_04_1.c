type_init(serial_register_types)



static bool serial_isa_init(ISABus *bus, int index, CharDriverState *chr)

{

    DeviceState *dev;

    ISADevice *isadev;



    isadev = isa_try_create(bus, TYPE_ISA_SERIAL);

    if (!isadev) {

        return false;

    }

    dev = DEVICE(isadev);

    qdev_prop_set_uint32(dev, "index", index);

    qdev_prop_set_chr(dev, "chardev", chr);

    if (qdev_init(dev) < 0) {

        return false;

    }

    return true;

}
