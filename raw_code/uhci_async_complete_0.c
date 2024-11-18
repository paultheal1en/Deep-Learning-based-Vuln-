static void uhci_async_complete(USBPacket *packet, void *opaque)

{

    UHCIState *s = opaque;

    UHCIAsync *async = (UHCIAsync *) packet;



    DPRINTF("uhci: async complete. td 0x%x token 0x%x\n", async->td, async->token);



    async->done = 1;



    uhci_process_frame(s);

}
