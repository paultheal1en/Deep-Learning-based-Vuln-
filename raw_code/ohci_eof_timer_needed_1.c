static bool ohci_eof_timer_needed(void *opaque)

{

    OHCIState *ohci = opaque;



    return ohci->eof_timer != NULL;

}
