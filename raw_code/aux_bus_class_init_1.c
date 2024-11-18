static void aux_bus_class_init(ObjectClass *klass, void *data)

{

    BusClass *k = BUS_CLASS(klass);



    /* AUXSlave has an MMIO so we need to change the way we print information

     * in monitor.

     */

    k->print_dev = aux_slave_dev_print;

}
