int i2c_recv(I2CBus *bus)

{

    I2CSlaveClass *sc;



    if ((QLIST_EMPTY(&bus->current_devs)) || (bus->broadcast)) {

        return -1;

    }



    sc = I2C_SLAVE_GET_CLASS(QLIST_FIRST(&bus->current_devs)->elt);

    if (sc->recv) {

        return sc->recv(QLIST_FIRST(&bus->current_devs)->elt);

    }

    return -1;

}
