int i2c_send(I2CBus *bus, uint8_t data)

{

    I2CSlaveClass *sc;

    I2CNode *node;

    int ret = 0;



    QLIST_FOREACH(node, &bus->current_devs, next) {

        sc = I2C_SLAVE_GET_CLASS(node->elt);

        if (sc->send) {

            ret = ret || sc->send(node->elt, data);

        } else {

            ret = -1;

        }

    }

    return ret ? -1 : 0;

}
