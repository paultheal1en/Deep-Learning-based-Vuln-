static void caps_to_network(RDMACapabilities *cap)

{

    cap->version = htonl(cap->version);

    cap->flags = htonl(cap->flags);

}
