static int ohci_bus_start(OHCIState *ohci)

{

    ohci->eof_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                    ohci_frame_boundary,

                    ohci);



    if (ohci->eof_timer == NULL) {

        trace_usb_ohci_bus_eof_timer_failed(ohci->name);

        ohci_die(ohci);

        return 0;

    }



    trace_usb_ohci_start(ohci->name);



    /* Delay the first SOF event by one frame time as

     * linux driver is not ready to receive it and

     * can meet some race conditions

     */



    ohci_eof_timer(ohci);



    return 1;

}
