static int i2c_slave_post_load(void *opaque, int version_id)

{

    I2CSlave *dev = opaque;

    I2CBus *bus;

    I2CNode *node;



    bus = I2C_BUS(qdev_get_parent_bus(DEVICE(dev)));

    if ((bus->saved_address == dev->address) || (bus->broadcast)) {

        node = g_malloc(sizeof(struct I2CNode));

        node->elt = dev;

        QLIST_INSERT_HEAD(&bus->current_devs, node, next);

    }

    return 0;

}
