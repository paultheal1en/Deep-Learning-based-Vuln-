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



    ohci_sof(ohci);



    return 1;

}
