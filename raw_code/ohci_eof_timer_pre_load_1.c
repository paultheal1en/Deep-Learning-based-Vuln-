static int ohci_eof_timer_pre_load(void *opaque)

{

    OHCIState *ohci = opaque;



    ohci_bus_start(ohci);



    return 0;

}
