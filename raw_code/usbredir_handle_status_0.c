static int usbredir_handle_status(USBRedirDevice *dev,

                                       int status, int actual_len)

{

    switch (status) {

    case usb_redir_success:

        return actual_len;

    case usb_redir_stall:

        return USB_RET_STALL;

    case usb_redir_cancelled:

        WARNING("returning cancelled packet to HC?\n");

        return USB_RET_NAK;

    case usb_redir_inval:

        WARNING("got invalid param error from usb-host?\n");

        return USB_RET_NAK;

    case usb_redir_babble:

        return USB_RET_BABBLE;

    case usb_redir_ioerror:

    case usb_redir_timeout:

    default:

        return USB_RET_IOERROR;

    }

}
