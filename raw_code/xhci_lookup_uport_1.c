static USBPort *xhci_lookup_uport(XHCIState *xhci, uint32_t *slot_ctx)

{

    USBPort *uport;

    char path[32];

    int i, pos, port;



    port = (slot_ctx[1]>>16) & 0xFF;




    port = xhci->ports[port-1].uport->index+1;

    pos = snprintf(path, sizeof(path), "%d", port);

    for (i = 0; i < 5; i++) {

        port = (slot_ctx[0] >> 4*i) & 0x0f;

        if (!port) {

            break;


        pos += snprintf(path + pos, sizeof(path) - pos, ".%d", port);




    QTAILQ_FOREACH(uport, &xhci->bus.used, next) {

        if (strcmp(uport->path, path) == 0) {

            return uport;



