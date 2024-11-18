static void ehci_update_frindex(EHCIState *ehci, int frames)

{

    int i;



    if (!ehci_enabled(ehci)) {

        return;

    }



    for (i = 0; i < frames; i++) {

        ehci->frindex += 8;



        if (ehci->frindex == 0x00002000) {

            ehci_raise_irq(ehci, USBSTS_FLR);

        }



        if (ehci->frindex == 0x00004000) {

            ehci_raise_irq(ehci, USBSTS_FLR);

            ehci->frindex = 0;

            if (ehci->usbsts_frindex >= 0x00004000) {

                ehci->usbsts_frindex -= 0x00004000;

            } else {

                ehci->usbsts_frindex = 0;

            }

        }

    }

}
